import { useState } from "react";

import axios from "axios";

import "./App.css";

import UploadBox from "./components/UploadBox";

import OutfitScoreCard from "./components/OutfitScoreCard";

import StyleCard from "./components/StyleCard";

import ClothingCard from "./components/ClothingCard";

import RecommendationCard from "./components/RecommendationCard";

import ColorPalette from "./components/ColorPalette";

import FashionNarrator from "./components/FashionNarrator";

import MetricsGrid from "./components/MetricsGrid";

import SegmentationPreview from "./components/SegmentationPreview";

import AestheticCard from "./components/AestheticCard";

import LuxuryCard from "./components/LuxuryCard";

import SneakerCard from "./components/SneakerCard";

import TextureCard from "./components/TextureCard";

import ComplexityCard from "./components/ComplexityCard";

import FashionAnalytics from "./components/FashionAnalytics";

import LayerAnalysisCard from "./components/LayerAnalysisCard";

import SemanticFashionCard from "./components/SemanticFashionCard";


function App() {

  const [image, setImage] =
    useState(null);

  const [preview, setPreview] =
    useState(null);

  const [loading, setLoading] =
    useState(false);

  const [result, setResult] =
    useState(null);


  // =========================
  // IMAGE UPLOAD
  // =========================

  const handleImageChange = (
    e
  ) => {

    const file =
      e.target.files[0];

    if (!file) return;

    setImage(file);

    setPreview(
      URL.createObjectURL(file)
    );
  };


  // =========================
  // ANALYZE OUTFIT
  // =========================

  const analyzeOutfit =
    async () => {

      if (!image) {

        alert(
          "Please upload an image"
        );

        return;
      }

      setLoading(true);

      const formData =
        new FormData();

      formData.append(
        "file",
        image
      );

      try {

        const response =
          await axios.post(

            "http://127.0.0.1:8000/upload",

            formData,

            {
              headers: {

                "Content-Type":
                  "multipart/form-data",
              },
            }
          );

        setResult(
          response.data
        );

      } catch (error) {

        console.error(error);

        alert(
          "Analysis failed. Check backend."
        );

      } finally {

        setLoading(false);
      }
    };


  // =========================
  // SAFE ACCESS
  // =========================

  const analysis =
    result?.analysis;

  const segmentation =
    result?.segmentation;

  const validation =
    result?.validation;


  return (

    <div className="container">

      {/* ========================= */}
      {/* HERO */}
      {/* ========================= */}

      <div className="hero-section">

        <h1>
          AI Fashion Intelligence
        </h1>

        <p className="subtitle">

          Advanced AI-Powered
          Outfit Analysis Platform

        </p>

      </div>


      {/* ========================= */}
      {/* UPLOAD */}
      {/* ========================= */}

      <UploadBox

        handleImageChange={
          handleImageChange
        }

        analyzeOutfit={
          analyzeOutfit
        }

        preview={preview}

        loading={loading}
      />


      {/* ========================= */}
      {/* LOADING */}
      {/* ========================= */}

      {
        loading && (

          <div className="card">

            <div className="loading-spinner">

            </div>

            <p className="loading-text">

              AI Fashion Engine
              Processing...

            </p>

          </div>
        )
      }


      {/* ========================= */}
      {/* RESULTS */}
      {/* ========================= */}

      {
        result && (

          <div className="dashboard">

            {/* ========================= */}
            {/* VALIDATION */}
            {/* ========================= */}

            {
              !validation
                ?.valid_image && (

                <div className="card error-card">

                  <h3>
                    Validation Errors
                  </h3>

                  {
                    validation
                      ?.validation_errors
                      ?.map(
                        (
                          error,
                          index
                        ) => (

                          <p key={index}>
                            ❌ {error}
                          </p>
                        )
                      )
                  }

                </div>
              )
            }


            {/* ========================= */}
            {/* MAIN DASHBOARD */}
            {/* ========================= */}

            {
              validation
                ?.valid_image && (

                <>

                  {/* ========================= */}
                  {/* TOP GRID */}
                  {/* ========================= */}

                  <div className="top-grid">

                    <OutfitScoreCard
                      outfit={
                        analysis
                          ?.outfit_analysis
                      }
                    />

                    <StyleCard
                      style={
                        analysis
                          ?.fashion_style_analysis
                      }
                    />

                    <AestheticCard
                      aesthetic={
                        analysis
                          ?.fashion_aesthetic
                      }
                    />

                    <LuxuryCard
                      luxury={
                        analysis
                          ?.luxury_analysis
                      }
                    />

                  </div>


                  {/* ========================= */}
                  {/* SECOND GRID */}
                  {/* ========================= */}

                  <div className="top-grid">

                    <ClothingCard
                      clothing={
                        analysis
                          ?.clothing_analysis
                      }
                    />

                    <SneakerCard
                      sneaker={
                        analysis
                          ?.sneaker_analysis
                      }
                    />

                    <TextureCard
                      texture={
                        analysis
                          ?.texture_analysis
                      }
                    />

                    <ComplexityCard
                      complexity={
                        analysis
                          ?.outfit_complexity
                      }
                    />

                    <LayerAnalysisCard

                      layer={
                        analysis
                        ?.layer_analysis
                      }
                    />

                    <SemanticFashionCard

                      semantic={
                        analysis
                        ?.semantic_fashion_analysis
                      }
                    />

                  </div>


                  {/* ========================= */}
                  {/* COLOR ANALYSIS */}
                  {/* ========================= */}

                  {
                    analysis
                      ?.color_analysis && (

                      <ColorPalette

                        colors={
                          analysis
                            ?.color_analysis
                            ?.dominant_colors
                        }
                      />
                    )
                  }


                  {/* ========================= */}
                  {/* SEGMENTATION */}
                  {/* ========================= */}

                  {
                    segmentation
                      ?.fashion_segmentation && (

                      <SegmentationPreview

                        segmentation={
                          segmentation
                            ?.fashion_segmentation
                        }

                        background={
                          segmentation
                            ?.background_removal
                            ?.output_path
                        }
                      />
                    )
                  }


                  {/* ========================= */}
                  {/* FASHION METRICS */}
                  {/* ========================= */}

                  {
                    analysis
                      ?.fashion_metrics && (

                      <MetricsGrid

                        metrics={
                          analysis
                            ?.fashion_metrics
                        }
                      />
                    )
                  }


                  {/* ========================= */}
                  {/* AI ANALYTICS */}
                  {/* ========================= */}

                  {
                    analysis
                      ?.fashion_metrics && (

                      <FashionAnalytics

                        metrics={
                          analysis
                            ?.fashion_metrics
                        }

                        luxury={
                          analysis
                            ?.luxury_analysis
                        }

                        complexity={
                          analysis
                            ?.outfit_complexity
                        }
                      />
                    )
                  }


                  {/* ========================= */}
                  {/* AI NARRATOR */}
                  {/* ========================= */}

                  {
                    analysis
                      ?.fashion_narration && (

                      <FashionNarrator

                        narration={
                          analysis
                            ?.fashion_narration
                            ?.fashion_narration
                        }
                      />
                    )
                  }


                  {/* ========================= */}
                  {/* RECOMMENDATIONS */}
                  {/* ========================= */}

                  {
                    analysis
                      ?.style_recommendations
                      ?.style_recommendations && (

                      <RecommendationCard

                        recommendations={
                          analysis
                            ?.style_recommendations
                            ?.style_recommendations
                        }
                      />
                    )
                  }

                </>
              )
            }

          </div>
        )
      }

    </div>
  );
}

export default App;