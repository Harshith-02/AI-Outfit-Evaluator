function ComplexityCard({

  complexity
}) {

  if (!complexity) return null;

  return (

    <div className="card">

      <h3>
        Outfit Complexity
      </h3>

      <p>

        <strong>
          Level:
        </strong>

        {" "}

        {complexity.complexity_level}

      </p>

      <p>

        <strong>
          Complexity Score:
        </strong>

        {" "}

        {complexity.complexity_score}%

      </p>

    </div>
  );
}

export default ComplexityCard;