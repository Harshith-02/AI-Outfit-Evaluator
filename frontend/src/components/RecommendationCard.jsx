function RecommendationCard({

  recommendations

}) {

  return (

    <div className="card">

      <h3>
        AI Style Recommendations
      </h3>

      <div className="recommendation-list">

        {
          recommendations.map(
            (
              item,
              index
            ) => (

              <div
                key={index}
                className="recommendation-item"
              >

                ✨ {item}

              </div>
            )
          )
        }

      </div>

    </div>
  );
}

export default RecommendationCard;