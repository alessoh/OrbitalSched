import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "OrbitalSched",
  description: "Thermal- and orbit-aware inference scheduler for orbital data centers",
};

export default function RootLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <html lang="en">
      <body>{children}</body>
    </html>
  );
}
