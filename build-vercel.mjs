import { cp, mkdir, rm, writeFile } from "node:fs/promises";
import path from "node:path";

const sourceStaticDir = path.resolve("app/static");
const outputStaticDir = path.resolve("static");

const apiBase = (process.env.API_BASE_URL || process.env.BACKEND_URL || "").trim();
const wsBase = (process.env.WS_BASE_URL || "").trim();

const config = {
  API_BASE_URL: apiBase,
  WS_BASE_URL: wsBase,
};

const content = [
  `window.SICHER_CONFIG = ${JSON.stringify(config, null, 2)};`,
  `window.sicherApiUrl = function(path) {`,
  `  const base = (window.SICHER_CONFIG && window.SICHER_CONFIG.API_BASE_URL) || "";`,
  `  return base ? base.replace(/\\/$/, "") + path : path;`,
  `};`,
  `window.sicherWsUrl = function(path) {`,
  `  const config = window.SICHER_CONFIG || {};`,
  `  const base = config.WS_BASE_URL || config.API_BASE_URL || "";`,
  `  if (!base) {`,
  `    const proto = window.location.protocol === "https:" ? "wss" : "ws";`,
  `    return proto + "://" + window.location.host + path;`,
  `  }`,
  `  const url = new URL(base);`,
  `  const proto = url.protocol === "https:" ? "wss:" : "ws:";`,
  `  return proto + "//" + url.host + path;`,
  `};`,
  "",
].join("\n");

await rm(outputStaticDir, { recursive: true, force: true });
await cp(sourceStaticDir, outputStaticDir, { recursive: true });
await writeFile(path.join(outputStaticDir, "config.js"), content, "utf8");

if (!apiBase) {
  console.warn("API_BASE_URL is empty; frontend will fall back to same-origin requests.");
}
