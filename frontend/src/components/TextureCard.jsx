function TextureCard({

  texture
}) {

  if (!texture) return null;

  return (

    <div className="card">

      <h3>
        Texture Analysis
      </h3>

      <p>

        <strong>
          Texture Type:
        </strong>

        {" "}

        {texture.texture_type}

      </p>

      <p>

        <strong>
          Texture Score:
        </strong>

        {" "}

        {texture.texture_score}%

      </p>

    </div>
  );
}

export default TextureCard;