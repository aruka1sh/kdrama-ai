import type { Metadata } from "next";
import Insights from "@/components/Insights";
import { PageHeader } from "@/components/ui";

export const metadata: Metadata = { title: "AI Insights" };

export default function InsightsPage() {
  return (
    <>
      <PageHeader eyebrow="Project 2 · Unsupervised Learning" title="Explore the Hidden Patterns of K-Dramas" korean="패턴">
        No labels, no right answers. Six algorithms look at the same five numbers for every drama and show us
        groups, simple structure and the titles that stand apart.
      </PageHeader>
      <Insights />
    </>
  );
}
