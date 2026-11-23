import type { Metadata } from "next";
import Predictor from "@/components/Predictor";
import { PageHeader } from "@/components/ui";

export const metadata: Metadata = { title: "AI Predictor" };

export default function PredictorPage() {
  return (
    <>
      <PageHeader eyebrow="Project 1 · Supervised Learning" title="Will You Love This K-Drama?" korean="예측">
        Describe a drama and your taste. A scikit-learn model trained on real MyDramaList reviews estimates the
        chance that a viewer like you would score it 8 or higher.
      </PageHeader>
      <Predictor />
    </>
  );
}
