function ColorPalette({ colors }) {

  return (

    <div className="card">

      <h3>Detected Colors</h3>

      <div className="palette">

        {
          colors.map(
            (color, index) => (

              <div
                key={index}
                className="color-box"
              >

                <div
                  className="swatch"
                  style={{
                    background: color
                  }}
                />

                <p>{color}</p>

              </div>
            )
          )
        }

      </div>

    </div>
  );
}

export default ColorPalette;