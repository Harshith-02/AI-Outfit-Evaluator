import ConfidenceBar from "./ConfidenceBar";

function MetricsGrid({
  metrics
}) {

  return (

    <div className="card">

      <h3>
        Fashion Intelligence
      </h3>

      <ConfidenceBar
        label="Color Harmony"
        value={
          metrics.color_harmony
        }
      />

      <ConfidenceBar
        label="Contrast Balance"
        value={
          metrics.contrast_balance
        }
      />

      <ConfidenceBar
        label="Formality Score"
        value={
          metrics.formality_score
        }
      />

      <ConfidenceBar
        label="Modern Style"
        value={
          metrics.modern_style_score
        }
      />

    </div>
  );
}

export default MetricsGrid;