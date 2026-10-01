import io
from typing import Dict, Any, Tuple
from PIL import Image, UnidentifiedImageError

SUPPORTED_FORMATS = {"JPEG", "PNG", "WEBP"}
SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}
SUPPORTED_MIME_TYPES = {"image/jpeg", "image/png", "image/webp"}

# Constraints
MAX_FILE_SIZE_BYTES = 10 * 1024 * 1024  # 10 MB
MIN_DIMENSION = 10
MAX_DIMENSION = 8000
MAX_ASPECT_RATIO = 100.0


class ImageValidationError(Exception):
    """Custom exception for image validation failures."""

    def __init__(self, code: str, message: str, status_code: int = 400):
        self.code = code
        self.message = message
        self.status_code = status_code
        super().__init__(message)


class ImageProcessingService:
    """Service for validating, safely decoding, and inspecting images.
    Operates strictly in-memory without persistent disk storage or ML dependencies.
    """

    @staticmethod
    def validate_and_inspect_image(
        file_bytes: bytes,
        filename: str,
        content_type: str = None,
    ) -> Dict[str, Any]:
        """Validate an uploaded image and extract structured metadata.

        Args:
            file_bytes: Raw binary content of the uploaded file.
            filename: Original name of the uploaded file.
            content_type: MIME type reported in the upload header.

        Returns:
            Structured dictionary containing validation status and image metadata.

        Raises:
            ImageValidationError: If validation fails at any stage.
        """
        # 1. Check presence and non-emptiness
        if not file_bytes or len(file_bytes) == 0:
            raise ImageValidationError(
                code="EMPTY_FILE",
                message="Uploaded file is empty.",
            )

        size_bytes = len(file_bytes)

        # 2. Check maximum file size
        if size_bytes > MAX_FILE_SIZE_BYTES:
            max_mb = MAX_FILE_SIZE_BYTES // (1024 * 1024)
            raise ImageValidationError(
                code="FILE_TOO_LARGE",
                message=f"Image size exceeds the maximum allowed limit of {max_mb}MB.",
            )

        # 3. MIME type inspection if provided
        if content_type:
            clean_mime = content_type.split(";")[0].strip().lower()
            if clean_mime not in SUPPORTED_MIME_TYPES:
                # Some clients might send application/octet-stream; we will verify format via binary header,
                # but if an explicitly unsupported image MIME like image/gif is sent, reject immediately.
                if clean_mime.startswith("image/") or clean_mime in {"application/pdf", "text/plain"}:
                    raise ImageValidationError(
                        code="UNSUPPORTED_MIME_TYPE",
                        message=f"Unsupported media type: '{clean_mime}'. Supported formats are JPG, JPEG, PNG, and WEBP.",
                    )

        # 4. Binary header / integrity inspection with Pillow
        stream = io.BytesIO(file_bytes)
        try:
            with Image.open(stream) as img:
                detected_format = (img.format or "").upper()
                if detected_format not in SUPPORTED_FORMATS:
                    raise ImageValidationError(
                        code="UNSUPPORTED_FORMAT",
                        message=f"Unsupported image format: '{detected_format or 'UNKNOWN'}'. Supported formats are JPG, JPEG, PNG, and WEBP.",
                    )
                # Verify header integrity
                img.verify()
        except UnidentifiedImageError:
            raise ImageValidationError(
                code="UNIDENTIFIED_IMAGE",
                message="File is not a valid image or the format could not be decoded.",
            )
        except ImageValidationError:
            raise
        except Exception as e:
            raise ImageValidationError(
                code="CORRUPTED_IMAGE",
                message="Image file is corrupted and could not be verified.",
            )

        # 5. Full raster decode, dimension and RGB conversion test
        stream.seek(0)
        try:
            with Image.open(stream) as img:
                width, height = img.size
                actual_format = img.format

                # Check dimensions
                if width < MIN_DIMENSION or height < MIN_DIMENSION:
                    raise ImageValidationError(
                        code="DIMENSIONS_TOO_SMALL",
                        message=f"Image dimensions ({width}x{height}) are smaller than minimum allowed ({MIN_DIMENSION}x{MIN_DIMENSION}).",
                    )

                if width > MAX_DIMENSION or height > MAX_DIMENSION:
                    raise ImageValidationError(
                        code="DIMENSIONS_TOO_LARGE",
                        message=f"Image dimensions ({width}x{height}) exceed maximum allowed limit of ({MAX_DIMENSION}x{MAX_DIMENSION}).",
                    )

                # Check aspect ratio
                aspect_ratio = max(width / height, height / width)
                if aspect_ratio > MAX_ASPECT_RATIO:
                    raise ImageValidationError(
                        code="UNUSUAL_ASPECT_RATIO",
                        message=f"Image aspect ratio ({aspect_ratio:.1f}) exceeds the acceptable threshold.",
                    )

                # Test RGB conversion capability and trigger full raster decompression
                try:
                    rgb_img = img.convert("RGB")
                    rgb_img.load()
                except Exception:
                    raise ImageValidationError(
                        code="RGB_CONVERSION_FAILED",
                        message="Image raster data is incomplete or cannot be converted to standard RGB.",
                    )

                return {
                    "valid": True,
                    "filename": filename or "uploaded_image",
                    "format": actual_format,
                    "width": width,
                    "height": height,
                    "size_bytes": size_bytes,
                }
        except ImageValidationError:
            raise
        except Exception:
            raise ImageValidationError(
                code="CORRUPTED_IMAGE",
                message="Image could not be fully decoded or raster data is corrupt.",
            )


image_service = ImageProcessingService()
