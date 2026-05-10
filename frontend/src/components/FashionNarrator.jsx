function FashionNarrator({
  narration
}) {

  return (

    <div className="card narrator-card">

      <h3>AI Fashion Narrator</h3>

      <p className="narration-text">
        {narration}
      </p>

    </div>
  );
}

export default FashionNarrator;