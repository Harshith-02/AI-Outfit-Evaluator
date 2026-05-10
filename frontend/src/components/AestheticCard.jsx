function AestheticCard({
  aesthetic
}) {

  return (

    <div className="card aesthetic-card">

      <h3>
        Fashion Aesthetic
      </h3>

      <div className="aesthetic-badge">

        {
          aesthetic
            .fashion_aesthetic
        }

      </div>

    </div>
  );
}

export default AestheticCard;