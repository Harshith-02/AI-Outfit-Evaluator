import {

  RadarChart,

  PolarGrid,

  PolarAngleAxis,

  PolarRadiusAxis,

  Radar,

  ResponsiveContainer

} from "recharts";


function FashionAnalytics({

  metrics,

  luxury,

  complexity

}) {

  if (
    !metrics ||
    !luxury ||
    !complexity
  ) return null;

  const data = [

    {
      metric:
        "Harmony",

      value:
        metrics.color_harmony
    },

    {
      metric:
        "Contrast",

      value:
        metrics.contrast_balance
    },

    {
      metric:
        "Formality",

      value:
        metrics.formality_score
    },

    {
      metric:
        "Modern",

      value:
        metrics.modern_style_score
    },

    {
      metric:
        "Luxury",

      value:
        luxury.luxury_score
    },

    {
      metric:
        "Complexity",

      value:
        complexity.complexity_score
    }
  ];

  return (

    <div className="card analytics-card">

      <h3>
        AI Fashion Analytics
      </h3>

      <div
        style={{
          width: "100%",
          height: 400
        }}
      >

        <ResponsiveContainer>

          <RadarChart
            data={data}
          >

            <PolarGrid />

            <PolarAngleAxis
              dataKey="metric"
            />

            <PolarRadiusAxis
              angle={30}
              domain={[0, 100]}
            />

            <Radar

              name="Fashion"

              dataKey="value"

              stroke="#c084fc"

              fill="#a855f7"

              fillOpacity={0.5}
            />

          </RadarChart>

        </ResponsiveContainer>

      </div>

    </div>
  );
}

export default FashionAnalytics;