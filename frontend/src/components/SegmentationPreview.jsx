function SegmentationPreview({

  segmentation,

  background

}) {

  if (!segmentation) return null;

  return (

    <div className="card">

      <h3>
        AI Fashion Segmentation
      </h3>

      {
        background && (

          <div className="segmented-preview">

            <img
              src={`http://127.0.0.1:8000/${background}`}
              alt="segmented"
              className="segmented-image"
            />

          </div>
        )
      }

      <div className="segmentation-tags">

        {
          segmentation
            .detected_clothing_items
            .map(
              (
                item,
                index
              ) => (

                <span
                  key={index}
                  className="tag"
                >

                  {item}

                </span>
              )
            )
        }

      </div>

    </div>
  );
}

export default SegmentationPreview;