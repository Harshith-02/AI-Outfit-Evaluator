function OutfitScoreCard({ outfit }) {

  return (

    <div className="card">

      <h3>Outfit Score</h3>

      <div className="score-circle">

        {outfit.outfit_score}

      </div>

      <p>
        {outfit.style}
      </p>

      <p>
        Confidence:
        {" "}
        {outfit.confidence_score}%
      </p>

    </div>
  );
}

export default OutfitScoreCard;