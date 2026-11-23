import type { Metadata } from "next";
import About from "@/components/About";
import { PageHeader } from "@/components/ui";

export const metadata: Metadata = { title: "About the Project" };

export default function AboutPage() {
  return (
    <>
      <PageHeader eyebrow="About the Project" title="Two machine-learning projects, one K-Drama story" korean="이야기">
        What we built, where the data comes from, how we evaluated it, and what it can and cannot tell us.
      </PageHeader>
      <About />
    </>
  );
}
