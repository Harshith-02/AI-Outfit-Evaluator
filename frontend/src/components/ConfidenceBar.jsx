function ConfidenceBar({

  label,

  value

}) {

  return (

    <div className="confidence-container">

      <div className="confidence-header">

        <span>{label}</span>

        <span>{value}%</span>

      </div>

      <div className="confidence-track">

        <div
          className="confidence-fill"
          style={{
            width: `${value}%`
          }}
        />

      </div>

    </div>
  );
}

export default ConfidenceBar;