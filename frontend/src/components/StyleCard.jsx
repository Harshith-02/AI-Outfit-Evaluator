function StyleCard({ style }) {

  return (

    <div className="card">

      <h3>Detected Style</h3>

      <p className="style-text">
        {style.detected_style}
      </p>

      <p>
        Confidence:
        {" "}
        {style.confidence}%
      </p>

    </div>
  );
}

export default StyleCard;