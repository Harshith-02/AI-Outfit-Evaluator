function LayerAnalysisCard({

  layer
}) {

  if (!layer) return null;

  return (

    <div className="card">

      <h3>
        Layer Intelligence
      </h3>

      <p>

        <strong>
          Primary Outerwear:
        </strong>

        {" "}

        {layer.primary_outerwear}

      </p>

      <p>

        <strong>
          Inner Layer:
        </strong>

        {" "}

        {layer.inner_layer}

      </p>

      <p>

        <strong>
          Layer Style:
        </strong>

        {" "}

        {layer.layer_style}

      </p>

    </div>
  );
}

export default LayerAnalysisCard;