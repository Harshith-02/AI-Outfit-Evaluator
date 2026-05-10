function SemanticFashionCard({

  semantic
}) {

  if (!semantic) return null;

  return (

    <div className="card semantic-card">

      <h3>
        Semantic Fashion AI
      </h3>

      <div className="semantic-grid">

        <div className="semantic-item">

          <span>
            Semantic Outerwear
          </span>

          <h4>
            {
              semantic
              ?.semantic_outerwear
            }
          </h4>

        </div>


        <div className="semantic-item">

          <span>
            Semantic Footwear
          </span>

          <h4>
            {
              semantic
              ?.semantic_footwear
            }
          </h4>

        </div>


        <div className="semantic-item">

          <span>
            Fit Type
          </span>

          <h4>
            {
              semantic
              ?.fit_type
            }
          </h4>

        </div>


        <div className="semantic-item">

          <span>
            Fashion Identity
          </span>

          <h4>
            {
              semantic
              ?.fashion_identity
            }
          </h4>

        </div>

      </div>

    </div>
  );
}

export default SemanticFashionCard;