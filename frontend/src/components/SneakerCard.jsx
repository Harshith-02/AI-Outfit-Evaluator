function SneakerCard({

  sneaker
}) {

  if (!sneaker) return null;

  return (

    <div className="card">

      <h3>
        Sneaker Intelligence
      </h3>

      <p>

        <strong>
          Type:
        </strong>

        {" "}

        {sneaker.sneaker_type}

      </p>

      <p>

        <strong>
          Sneaker Score:
        </strong>

        {" "}

        {sneaker.sneaker_score}%

      </p>

    </div>
  );
}

export default SneakerCard;