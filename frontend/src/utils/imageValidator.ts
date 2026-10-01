import type { ClientImageMetadata } from '../types';

export const MAX_FILE_SIZE_BYTES = 10 * 1024 * 1024; // 10MB
export const MIN_DIMENSION = 10;
export const MAX_DIMENSION = 8000;

const SUPPORTED_EXTENSIONS = ['.jpg', '.jpeg', '.png', '.webp'];
const SUPPORTED_MIME_TYPES = ['image/jpeg', 'image/png', 'image/webp'];

export function formatBytes(bytes: number, decimals = 2): string {
  if (bytes === 0) return '0 Bytes';
  const k = 1024;
  const dm = decimals < 0 ? 0 : decimals;
  const sizes = ['Bytes', 'KB', 'MB', 'GB'];
  const i = Math.floor(Math.log(bytes) / Math.log(k));
  return `${parseFloat((bytes / Math.pow(k, i)).toFixed(dm))} ${sizes[i]}`;
}

export function detectFormatFromFilename(filename: string): string {
  const ext = filename.slice(((filename.lastIndexOf('.') - 1) >>> 0) + 2).toLowerCase();
  switch (ext) {
    case 'jpg':
    case 'jpeg':
      return 'JPEG';
    case 'png':
      return 'PNG';
    case 'webp':
      return 'WEBP';
    default:
      return ext.toUpperCase() || 'UNKNOWN';
  }
}

/**
 * Performs strict client-side validation on an image file.
 * Validates presence, extension, MIME type, file size, and decodes the image
 * in the browser to extract dimensions and verify raster integrity.
 */
export async function validateImageClientSide(file: File): Promise<ClientImageMetadata> {
  // 1. File existence
  if (!file) {
    throw new Error('No image file selected.');
  }

  // 2. Empty file check
  if (file.size === 0) {
    throw new Error('Image file is empty.');
  }

  // 3. Extension check
  const ext = `.${file.name.split('.').pop()?.toLowerCase()}`;
  if (!SUPPORTED_EXTENSIONS.includes(ext)) {
    throw new Error(
      `Unsupported image format (${ext}). Supported formats: JPG, JPEG, PNG, WEBP.`
    );
  }

  // 4. MIME type check
  if (file.type && !SUPPORTED_MIME_TYPES.includes(file.type)) {
    throw new Error(
      `Unsupported media type (${file.type}). Supported formats: JPG, JPEG, PNG, WEBP.`
    );
  }

  // 5. Size check
  if (file.size > MAX_FILE_SIZE_BYTES) {
    throw new Error(
      `Image exceeds the maximum allowed size of ${formatBytes(MAX_FILE_SIZE_BYTES)}.`
    );
  }

  // 6. Actual image decoding check via HTMLImageElement
  const previewUrl = URL.createObjectURL(file);

  return new Promise((resolve, reject) => {
    const img = new Image();

    img.onload = () => {
      const width = img.naturalWidth;
      const height = img.naturalHeight;

      // 7. Dimension validation
      if (width < MIN_DIMENSION || height < MIN_DIMENSION) {
        URL.revokeObjectURL(previewUrl);
        reject(
          new Error(
            `Invalid image dimensions (${width}x${height}). Minimum dimension is ${MIN_DIMENSION}x${MIN_DIMENSION} pixels.`
          )
        );
        return;
      }

      if (width > MAX_DIMENSION || height > MAX_DIMENSION) {
        URL.revokeObjectURL(previewUrl);
        reject(
          new Error(
            `Invalid image dimensions (${width}x${height}). Maximum dimension is ${MAX_DIMENSION}x${MAX_DIMENSION} pixels.`
          )
        );
        return;
      }

      resolve({
        file,
        previewUrl,
        filename: file.name,
        format: detectFormatFromFilename(file.name),
        sizeBytes: file.size,
        width,
        height,
        aspectRatio: Number((width / height).toFixed(2)),
      });
    };

    img.onerror = () => {
      URL.revokeObjectURL(previewUrl);
      reject(new Error('Image could not be decoded. File may be corrupted or not a valid image.'));
    };

    img.src = previewUrl;
  });
}
