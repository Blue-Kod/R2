const { JSDOM } = require("jsdom");
const { createHash } = require("crypto");

const USER_AGENT =
  "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/138.0.0.0 Safari/537.36";
const VQD_STACK =
  "l@https://duck.ai/dist/duckai-dist/entry.duckai.c9340e95bd2f7fdc3302.js:2:1308110\n";

function computeVqd(challengeBase64) {
  const jsScript = Buffer.from(challengeBase64, "base64").toString("utf-8");
  const dom = new JSDOM(
    `<iframe id="jsa" sandbox="allow-scripts allow-same-origin" srcdoc="<!DOCTYPE html>
<html>
<head>
<meta http-equiv="Content-Security-Policy"; content="default-src 'none'; script-src 'unsafe-inline'">
</head>
<body></body>
</html>" style="position: absolute; left: -9999px; top: -9999px;"></iframe>`,
    { runScripts: "dangerously" }
  );
  dom.window.top.__DDG_BE_VERSION__ = 1;
  dom.window.top.__DDG_FE_CHAT_HASH__ = 1;
  const jsa = dom.window.top.document.querySelector("#jsa");
  const contentDoc = jsa.contentDocument || jsa.contentWindow.document;
  const meta = contentDoc.createElement("meta");
  meta.setAttribute("http-equiv", "Content-Security-Policy");
  meta.setAttribute("content", "default-src 'none'; script-src 'unsafe-inline';");
  contentDoc.head.appendChild(meta);

  return dom.window.eval(jsScript).then((result) => {
    result.client_hashes[0] = USER_AGENT;
    result.client_hashes = result.client_hashes.map((t) => {
      const hash = createHash("sha256");
      hash.update(t);
      return hash.digest("base64");
    });

    if (result.meta && typeof result.meta === "object") {
      result.meta.origin = "https://duck.ai";
      result.meta.stack = VQD_STACK;
      result.meta.duration = String(20 + Math.floor(Math.random() * 30));
    }
    return Buffer.from(JSON.stringify(result)).toString("base64");
  });
}

let input = "";
process.stdin.setEncoding("utf-8");
process.stdin.on("data", (chunk) => {
  input += chunk;
});
process.stdin.on("end", () => {
  const challenge = input.trim();
  if (!challenge) {
    process.stdout.write(JSON.stringify({ error: "no challenge on stdin" }));
    process.exit(1);
  }
  computeVqd(challenge)
    .then((vqd) => {
      process.stdout.write(JSON.stringify({ vqd }));
      process.exit(0);
    })
    .catch((e) => {
      process.stdout.write(
        JSON.stringify({ error: String((e && e.stack) || e) })
      );
      process.exit(1);
    });
});
