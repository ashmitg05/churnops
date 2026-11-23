import type { Metadata } from "next";
import { GeistMono } from "geist/font/mono";
import { GeistSans } from "geist/font/sans";
import { Nav } from "@/components/nav";
import "./globals.css";

export const metadata: Metadata = {
  title: "ChurnOps",
  description: "Churn prediction with versioned data, gated promotion and drift monitoring.",
};

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en" className={`${GeistSans.variable} ${GeistMono.variable}`}>
      <body className="min-h-[100dvh] font-sans">
        <div className="bg-stage" aria-hidden />
        <div className="grain" aria-hidden />
        <div className="relative z-10">
          <Nav />
          <main className="mx-auto max-w-[1280px] px-4 pb-24 pt-8 md:px-8 md:pt-12">{children}</main>
        </div>
      </body>
    </html>
  );
}
