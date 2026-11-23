import { Suspense } from "react";
import DramaDetails from "@/components/DramaDetails";

export default function DramaPage() {
  return (
    <Suspense fallback={<div className="min-h-screen" />}>
      <DramaDetails />
    </Suspense>
  );
}
