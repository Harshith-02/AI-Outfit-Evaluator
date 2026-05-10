function UploadBox({
  handleImageChange,
  analyzeOutfit,
  preview,
  loading
}) {

  return (

    <div className="upload-box">

      <input
        type="file"
        accept="image/*"
        onChange={handleImageChange}
      />

      {
        preview && (
          <img
            src={preview}
            alt="preview"
            className="preview-image"
          />
        )
      }

      <button
        onClick={analyzeOutfit}
        disabled={loading}
      >
        {
          loading
            ? "Analyzing..."
            : "Analyze Outfit"
        }
      </button>

    </div>
  );
}

export default UploadBox;