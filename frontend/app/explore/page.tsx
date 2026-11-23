import type { Metadata } from "next";
import { Suspense } from "react";
import Explore from "@/components/Explore";

export const metadata: Metadata = { title: "Explore Dramas" };

export default function ExplorePage() {
  return (
    <Suspense fallback={<div className="min-h-screen" />}>
      <Explore />
    </Suspense>
  );
}
