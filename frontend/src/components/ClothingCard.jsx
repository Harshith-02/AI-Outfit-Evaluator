function ClothingCard({ clothing }) {

  return (

    <div className="card">

      <h3>Clothing Analysis</h3>

      <p>
        Upper Wear:
        {" "}
        <strong>
          {clothing.upper_wear}
        </strong>
      </p>

      <p>
        Lower Wear:
        {" "}
        <strong>
          {clothing.lower_wear}
        </strong>
      </p>

      <p>
        Footwear:
        {" "}
        <strong>
          {clothing.footwear}
        </strong>
      </p>

    </div>
  );
}

export default ClothingCard;