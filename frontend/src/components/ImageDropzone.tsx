import React, { useState, useRef } from 'react';

interface ImageDropzoneProps {
  onImageSelected: (file: File) => void;
  disabled?: boolean;
}

export const ImageDropzone: React.FC<ImageDropzoneProps> = ({ onImageSelected, disabled = false }) => {
  const [isDragOver, setIsDragOver] = useState(false);
  const fileInputRef = useRef<HTMLInputElement>(null);

  const handleDragOver = (e: React.DragEvent) => {
    e.preventDefault();
    e.stopPropagation();
    if (!disabled) {
      setIsDragOver(true);
    }
  };

  const handleDragLeave = (e: React.DragEvent) => {
    e.preventDefault();
    e.stopPropagation();
    setIsDragOver(false);
  };

  const handleDrop = (e: React.DragEvent) => {
    e.preventDefault();
    e.stopPropagation();
    setIsDragOver(false);

    if (disabled) return;

    if (e.dataTransfer.files && e.dataTransfer.files.length > 0) {
      const file = e.dataTransfer.files[0];
      onImageSelected(file);
    }
  };

  const handleFileInputChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files && e.target.files.length > 0) {
      const file = e.target.files[0];
      onImageSelected(file);
    }
    // Reset file input value so selecting the same file again triggers change
    if (fileInputRef.current) {
      fileInputRef.current.value = '';
    }
  };

  const triggerPicker = () => {
    if (!disabled && fileInputRef.current) {
      fileInputRef.current.click();
    }
  };

  const handleKeyDown = (e: React.KeyboardEvent) => {
    if ((e.key === 'Enter' || e.key === ' ') && !disabled) {
      e.preventDefault();
      triggerPicker();
    }
  };

  const handleLoadSample = async (samplePath: string, filename: string) => {
    if (disabled) return;
    try {
      const response = await fetch(samplePath);
      const blob = await response.blob();
      const file = new File([blob], filename, { type: blob.type || 'image/jpeg' });
      onImageSelected(file);
    } catch (err) {
      console.error('Failed to load sample image:', err);
    }
  };

  return (
    <div
      className={`tactical-dropzone ${isDragOver ? 'drag-active' : ''} ${disabled ? 'disabled' : ''}`}
      onDragOver={handleDragOver}
      onDragEnter={handleDragOver}
      onDragLeave={handleDragLeave}
      onDrop={handleDrop}
      onClick={triggerPicker}
      onKeyDown={handleKeyDown}
      role="button"
      tabIndex={0}
      aria-label="Upload reconnaissance image"
    >
      <input
        ref={fileInputRef}
        type="file"
        accept=".jpg,.jpeg,.png,.webp,image/jpeg,image/png,image/webp"
        onChange={handleFileInputChange}
        style={{ position: 'absolute', opacity: 0, width: '1px', height: '1px', pointerEvents: 'none' }}
        data-testid="file-input"
        id="recon-file-input"
      />

      {/* Tactical Reticle Corners */}
      <div className="reticle-corner top-left" />
      <div className="reticle-corner top-right" />
      <div className="reticle-corner bottom-left" />
      <div className="reticle-corner bottom-right" />

      <div className="dropzone-content">
        <div className="dropzone-icon-hud">
          <svg width="48" height="48" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5">
            <path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4" />
            <polyline points="17 8 12 3 7 8" />
            <line x1="12" y1="3" x2="12" y2="15" />
          </svg>
          <div className="icon-pulse-ring" />
        </div>

        <div className="dropzone-text-group">
          <h3 className="dropzone-title">SELECT OR DROP RECONNAISSANCE IMAGE</h3>
          <p className="dropzone-subtitle">
            Target optical feed ingestion for Phase 2 validation testing
          </p>
        </div>

        <div className="format-badges-container">
          <span className="format-badge">JPG</span>
          <span className="format-badge">JPEG</span>
          <span className="format-badge">PNG</span>
          <span className="format-badge">WEBP</span>
          <span className="size-limit-badge">MAX 10 MB</span>
        </div>

        <div className="dropzone-action-cluster" onClick={(e) => e.stopPropagation()}>
          <button
            type="button"
            className="tactical-browse-btn"
            onClick={triggerPicker}
          >
            BROWSE FILES
          </button>

          <div className="sample-quick-loads">
            <span className="sample-prompt">OR TEST FEED:</span>
            <button
              type="button"
              className="sample-pill-btn"
              onClick={() => handleLoadSample('/samples/tank.jpg', 'tank.jpg')}
              data-testid="load-tank-sample"
              aria-label="Load tank sample reconnaissance image"
            >
              TANK (JPEG)
            </button>
            <button
              type="button"
              className="sample-pill-btn"
              onClick={() => handleLoadSample('/samples/ship.png', 'ship.png')}
              data-testid="load-ship-sample"
              aria-label="Load ship PNG sample image"
            >
              SHIP (PNG)
            </button>
            <button
              type="button"
              className="sample-pill-btn"
              onClick={() => handleLoadSample('/samples/drone.webp', 'drone.webp')}
              data-testid="load-drone-sample"
              aria-label="Load drone WEBP sample image"
            >
              DRONE (WEBP)
            </button>
            <button
              type="button"
              className="sample-pill-btn"
              onClick={() => handleLoadSample('/samples/panorama_tank.jpg', 'panorama_tank.jpg')}
              data-testid="load-pano-sample"
              aria-label="Load unusual aspect ratio panorama image"
            >
              PANO (ASPECT)
            </button>
            <button
              type="button"
              className="sample-pill-btn"
              onClick={() => handleLoadSample('/samples/radar_grayscale.png', 'radar_grayscale.png')}
              data-testid="load-grayscale-sample"
              aria-label="Load grayscale radar scan image"
            >
              RADAR (GRAY)
            </button>
            <button
              type="button"
              className="sample-pill-btn danger"
              onClick={() => handleLoadSample('/samples/invalid_doc.pdf', 'report.pdf')}
              data-testid="load-invalid-sample"
              aria-label="Test invalid non-image file"
            >
              INVALID PDF
            </button>
          </div>
        </div>
      </div>
    </div>
  );
};
