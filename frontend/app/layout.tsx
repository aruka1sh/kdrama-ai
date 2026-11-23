import type { Metadata } from "next";
import { Suspense } from "react";
import { Instrument_Serif, Manrope, Noto_Sans_KR } from "next/font/google";
import "./globals.css";
import Nav from "@/components/Nav";
import Footer from "@/components/Footer";

const display = Instrument_Serif({ variable: "--font-display-face", subsets: ["latin"], weight: "400", style: ["normal", "italic"] });

const body = Manrope({ variable: "--font-body-face", subsets: ["latin"] });

const korean = Noto_Sans_KR({ variable: "--font-kr-face", weight: ["400", "700"], preload: false });

export const metadata: Metadata = {
  title: { default: "K-Drama AI — Your Next Obsession Starts Here", template: "%s · K-Drama AI" },
  description:
    "Discover Korean dramas, build your watchlist and ask a real machine-learning model whether you will love them.",
};

export default function RootLayout({ children }: LayoutProps<"/">) {
  return (
    <html lang="en" data-scroll-behavior="smooth" className={`${display.variable} ${body.variable} ${korean.variable} h-full antialiased`}>
      <body className="min-h-full flex flex-col">
        {/* Nav reads the current path; on dynamic routes (/drama/[id]) it streams in */}
        <Suspense fallback={<div className="fixed inset-x-0 top-0 z-50 h-16 bg-ink/90" />}>
          <Nav />
        </Suspense>
        <main className="flex-1">{children}</main>
        <Footer />
      </body>
    </html>
  );
}
