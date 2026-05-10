function LuxuryCard({

  luxury
}) {

  if (!luxury) return null;

  return (

    <div className="card luxury-card">

      <h3>
        Luxury Intelligence
      </h3>

      <div className="luxury-score">

        {luxury.luxury_score}

      </div>

      <h2>
        {luxury.luxury_level}
      </h2>

    </div>
  );
}

export default LuxuryCard;