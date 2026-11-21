"""Actual installed Edge product journeys. Only public/CC0 QA inputs are uploaded."""

import argparse, base64, json, os, subprocess, time
from pathlib import Path
import httpx
from websockets.sync.client import connect

ROOT = Path(__file__).resolve().parents[1]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--url", default="http://127.0.0.1:8020")
    p.add_argument("--analysis-id", required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--inspect-only", action="store_true")
    a = p.parse_args()
    out = a.output.resolve()
    if not out.is_relative_to(ROOT / ".local"):
        raise ValueError("Browser evidence must remain private")
    out.mkdir(parents=True, exist_ok=True)
    url = a.url.rstrip("/")
    http = httpx.Client(timeout=60, trust_env=False)
    saved = http.get(url + "/api/analysis/" + a.analysis_id).json()
    edge = (
        Path(os.environ.get("PROGRAMFILES(X86)", "C:/Program Files (x86)"))
        / "Microsoft/Edge/Application/msedge.exe"
    )
    proc = subprocess.Popen(
        [
            str(edge),
            "--headless=new",
            "--disable-gpu",
            "--no-first-run",
            "--disable-sync",
            "--disable-background-networking",
            "--remote-debugging-address=127.0.0.1",
            "--remote-debugging-port=9229",
            "--no-proxy-server",
            "--host-resolver-rules=MAP * 0.0.0.0, EXCLUDE localhost, EXCLUDE 127.0.0.1",
            "--user-data-dir=" + str(out / "profile"),
            "about:blank",
        ],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
    )
    number = 0
    errors = []
    observations = []
    try:
        for _ in range(80):
            try:
                target = next(
                    t
                    for t in http.get("http://127.0.0.1:9229/json", timeout=1).json()
                    if t["type"] == "page"
                )
                break
            except (httpx.HTTPError, StopIteration):
                time.sleep(0.25)
        with connect(target["webSocketDebuggerUrl"], max_size=30_000_000) as ws:

            def cmd(method, params=None):
                nonlocal number
                number += 1
                ws.send(
                    json.dumps({"id": number, "method": method, "params": params or {}})
                )
                while True:
                    r = json.loads(ws.recv(timeout=30))
                    if r.get("method") == "Runtime.exceptionThrown":
                        errors.append(r["params"])
                    if r.get("id") == number:
                        if r.get("error"):
                            raise RuntimeError(r["error"])
                        return r.get("result", {})

            def js(expression):
                result = cmd(
                    "Runtime.evaluate",
                    {
                        "expression": expression,
                        "returnByValue": True,
                        "awaitPromise": True,
                    },
                )
                assert not result.get("exceptionDetails"), result
                return result["result"].get("value")

            def wait(expression, seconds=120):
                deadline = time.monotonic() + seconds
                while time.monotonic() < deadline:
                    value = js(expression)
                    if value:
                        return value
                    time.sleep(0.25)
                (out / "failure.json").write_text(
                    json.dumps(
                        {
                            "condition": expression,
                            "body": js("document.body.innerText"),
                            "errors": errors,
                        },
                        indent=2,
                    )
                )
                raise AssertionError("Browser condition timed out")

            def click(text, scope="document"):
                js(
                    f"Array.from({scope}.querySelectorAll('button')).find(b=>b.textContent.trim()==={json.dumps(text)}).click()"
                )

            def upload(selector, path):
                tree = cmd("DOM.getDocument")
                node = cmd(
                    "DOM.querySelector",
                    {"nodeId": tree["root"]["nodeId"], "selector": selector},
                )["nodeId"]
                cmd(
                    "DOM.setFileInputFiles",
                    {"nodeId": node, "files": [str(path.resolve())]},
                )

            def screenshot(name):
                (out / name).write_bytes(
                    base64.b64decode(
                        cmd("Page.captureScreenshot", {"format": "png"})["data"]
                    )
                )

            def open_report(identifier):
                cmd("Page.navigate", {"url": url + "?analysis=" + identifier})
                wait(
                    "document.querySelector('.results code')?.textContent==="
                    + json.dumps(identifier)
                )

            def upload_case(path, cloud=False, expect_error=False):
                click("New analysis", "document.querySelector('.main-nav')")
                wait(
                    "!!document.querySelector('input[aria-label=\"Submission document\"]')"
                )
                upload('input[aria-label="Submission document"]', path)
                wait(
                    "document.querySelector('.dropzone strong').textContent==="
                    + json.dumps(path.name)
                )
                js(
                    "(()=>{const c=document.querySelectorAll('.analysis-options input');if(c[0].checked!=="
                    + json.dumps(cloud)
                    + ")c[0].click();if(c[1].checked)c[1].click()})()"
                )
                js("document.querySelector('.upload-panel .primary-button').click()")
                wait(
                    "!!document.querySelector('.job-progress') || !!document.querySelector('.results') || !!document.querySelector('.global-error')"
                )
                assert not js("document.querySelector('.global-error')?.textContent"), (
                    js("document.body.innerText")
                )
                pending = js("new URL(location.href).searchParams.get('job')")
                if pending:
                    cmd("Page.reload")
                if expect_error:
                    wait("!!document.querySelector('.global-error')", 180)
                    return None
                identifier = wait(
                    "document.querySelector('.results code')?.textContent", 600
                )
                data = http.get(url + "/api/analysis/" + identifier).json()
                assert (
                    data["filename"] == path.name
                    and data["processing"]["options"]["cloud_review"] == cloud
                )
                assert not js("!!document.querySelector('.product-hero')")
                observations.append(
                    {
                        "filename": path.name,
                        "analysis_id": identifier,
                        "units": data["units_analyzed"],
                        "word_coverage": data["similarity_report"][
                            "textual_similarity_percent"
                        ],
                        "ai_status": data["semantic_assessment"]["status"],
                        "ai_findings": len(data["semantic_assessment"]["findings"]),
                        "job_refresh_reconnected": bool(pending),
                    }
                )
                return data

            cmd("Page.enable")
            cmd("Runtime.enable")
            cmd(
                "Emulation.setDeviceMetricsOverride",
                {
                    "width": 1440,
                    "height": 1050,
                    "deviceScaleFactor": 1,
                    "mobile": False,
                },
            )
            if a.inspect_only:
                open_report(a.analysis_id)
                wait('!!document.querySelector(".ai-finding")')
                screenshot("report-desktop.png")
                cmd(
                    "Emulation.setDeviceMetricsOverride",
                    {
                        "width": 390,
                        "height": 844,
                        "deviceScaleFactor": 1,
                        "mobile": True,
                    },
                )
                js(
                    "new Promise(r=>requestAnimationFrame(()=>requestAnimationFrame(r)))"
                )
                screenshot("report-mobile.png")
                widths = js(
                    "({viewport:innerWidth,body:document.documentElement.scrollWidth})"
                )
                assert widths["body"] <= widths["viewport"] + 1, widths
                click("Document", "document.querySelector('.report-tabs')")
                assert (
                    js("document.querySelector('.ai-document').textContent")
                    == saved["document"]["text"]
                )
                assert js("document.querySelectorAll('.ai-document mark').length") > 0
                assert not errors, errors
                (out / "inspection.json").write_text(
                    json.dumps(
                        {
                            "analysis_id": a.analysis_id,
                            "read_only": True,
                            "ai_findings": len(
                                saved["semantic_assessment"]["findings"]
                            ),
                            "document_preserved": True,
                            "highlights": True,
                            "mobile_widths": widths,
                            "javascript_exceptions": len(errors),
                        },
                        indent=2,
                    )
                    + "\n"
                )
                cmd("Browser.close")
                return
            # A completed job URL must reopen its report without resubmission.
            cmd("Page.navigate", {"url": url + "?job=" + saved["processing"]["job_id"]})
            wait(
                "document.querySelector('.results code')?.textContent==="
                + json.dumps(a.analysis_id)
            )
            assert js("document.querySelectorAll('.ai-finding').length") > 0
            cards = js(
                "Array.from(document.querySelectorAll('.ai-finding')).map(e=>({id:e.dataset.aiFinding,quotes:Array.from(e.querySelectorAll('.evidence-card blockquote')).map(b=>b.textContent)}))"
            )
            findings = {
                f["finding_id"]: f for f in saved["semantic_assessment"]["findings"]
            }
            for card in cards:
                assert card["quotes"] == [
                    findings[card["id"]]["submission_excerpt"],
                    findings[card["id"]]["source_excerpt"],
                ]
            screenshot("ai-report.png")
            click("Document", "document.querySelector('.report-tabs')")
            assert (
                js("document.querySelector('.ai-document').textContent")
                == saved["document"]["text"]
            )
            assert js("document.querySelectorAll('.ai-document mark').length") > 0
            click("Overview & AI review", "document.querySelector('.report-tabs')")
            assert js("document.body.innerText.includes('Investigation saved')")
            downloads = out / "downloads"
            downloads.mkdir(exist_ok=True)
            cmd(
                "Browser.setDownloadBehavior",
                {"behavior": "allow", "downloadPath": str(downloads)},
            )
            click("Download PDF")
            deadline = time.monotonic() + 60
            while (
                not (downloads / "sematrace-report.pdf").is_file()
                and time.monotonic() < deadline
            ):
                time.sleep(0.2)
            assert (downloads / "sematrace-report.pdf").read_bytes().startswith(b"%PDF")
            import pymupdf

            with pymupdf.open(downloads / "sematrace-report.pdf") as pdf:
                exported = "\n".join(page.get_text() for page in pdf)
                assert (
                    "AI-identified meaning correspondences" in exported
                    and a.analysis_id in exported
                )
            click('JSON')
            json_name=Path(saved['filename']).stem+'-sematrace-report.json'
            deadline=time.monotonic()+30
            while not (downloads/json_name).is_file() and time.monotonic()<deadline:
                time.sleep(.2)
            exported_json=json.loads((downloads/json_name).read_text(encoding='utf-8'))
            assert exported_json['analysis_id']==a.analysis_id
            assert exported_json['similarity_report']==saved['similarity_report']
            assert exported_json['semantic_assessment']==saved['semantic_assessment']
            # Real format uploads and local exclusion/restoration.
            exact = upload_case(ROOT / "demo/product-beta/01-exact.txt")
            assert exact["similarity_report"]["unique_matched_words"] > 0
            click("Matching words", "document.querySelector('.report-tabs')")
            js(
                "document.querySelector('.similarity-controls').open=true;document.querySelectorAll('.similarity-controls input[type=checkbox]').forEach(c=>{if(!c.checked)c.click()})"
            )
            js("new Promise(r=>requestAnimationFrame(()=>requestAnimationFrame(r)))")
            click(
                "Recalculate and save preliminary metric",
                "document.querySelector('.similarity-controls')",
            )
            wait(
                "parseFloat(document.querySelector('.similarity-heading h3').textContent)===0"
            )
            js(
                "document.querySelectorAll('.similarity-controls input[type=checkbox]').forEach(c=>{if(c.checked)c.click()})"
            )
            js("new Promise(r=>requestAnimationFrame(()=>requestAnimationFrame(r)))")
            click(
                "Recalculate and save preliminary metric",
                "document.querySelector('.similarity-controls')",
            )
            wait(
                "parseFloat(document.querySelector('.similarity-heading h3').textContent)==="
                + str(exact["similarity_report"]["textual_similarity_percent"])
            )
            assert (
                http.get(url + "/api/analysis/" + exact["analysis_id"]).json()[
                    "similarity_report"
                ]
                == exact["similarity_report"]
            )
            upload_case(ROOT / "demo/product-beta/01-exact.pdf")
            upload_case(ROOT / "demo/product-beta/01-exact.docx")
            modified=upload_case(ROOT / 'demo/product-beta/02-minor-edit.txt')
            assert 0<modified['similarity_report']['textual_similarity_percent']<100
            assert modified['findings']
            bad = out / "bad.docx"
            bad.write_bytes(b"not a document")
            upload_case(bad, expect_error=True)
            # Reference upload through the actual library UI.
            click(
                "Source library "
                + str(http.get(url + "/api/corpus").json()["source_origins"]),
                "document.querySelector('.main-nav')",
            )
            wait("!!document.querySelector('input[aria-label=\"Reference document\"]')")
            upload(
                'input[aria-label="Reference document"]',
                ROOT / "demo/sources/thermal-storage-reference.txt",
            )
            wait("!document.querySelector('.library-add .primary-button').disabled")
            js("document.querySelector('.library-add .primary-button').click()")
            wait(
                "document.querySelector('.library-add [role=status]')?.textContent", 300
            )
            screenshot("source-library.png")
            thermal = upload_case(
                ROOT / "demo/submissions/thermal-storage-paraphrased.txt", cloud=True
            )
            assert thermal["semantic_assessment"]["findings"]
            assert any(
                f["source_title"] == "thermal-storage-reference"
                for f in thermal["semantic_assessment"]["findings"]
            )
            screenshot("reference-paraphrase-report.png")
            original = out / "independent-original.txt"
            original.write_text(
                "At the Tuesday meeting, the Larkspur Chess Club decided to paint its storage cupboard orange and move the spare clocks into a locked drawer.",
                encoding="utf-8",
            )
            negative = upload_case(original, cloud=True)
            assert not negative["semantic_assessment"]["findings"]
            click("Saved reports", "document.querySelector('.main-nav')")
            wait("!!document.querySelector('.saved-reports[open]')")
            owned = exact["analysis_id"]
            selector = '.saved-report-row[data-analysis-id="' + owned + '"]'
            wait("!!document.querySelector(" + json.dumps(selector) + ")")
            click("Delete", f"document.querySelector({json.dumps(selector)})")
            wait(
                "Array.from(document.querySelector("
                + json.dumps(selector)
                + ').querySelectorAll("button")).some(b=>b.textContent==="Confirm deletion")'
            )
            click("Confirm deletion", f"document.querySelector({json.dumps(selector)})")
            wait("!document.querySelector(" + json.dumps(selector) + ")")
            assert http.get(url + "/api/analysis/" + owned).status_code == 404
            open_report(a.analysis_id)
            assert (
                http.get(url + "/api/analysis/" + a.analysis_id).json()[
                    "similarity_report"
                ]
                == saved["similarity_report"]
            )
            assert not errors, errors
            result = {
                "scope": "Actual single-origin local product UI, public/CC0 fixtures; functional checks, not independent accuracy",
                "saved_ai_report": a.analysis_id,
                "literal_ai_quotes": True,
                "document_highlights": True,
                "completed_job_resume": True,
                "actual_pdf_download": True,
                "actual_json_download": True,
                "modified_copy_checked": True,
                "source_upload_and_paraphrase": True,
                "malformed_upload_recovered": True,
                "word_exclusions_restored": True,
                "own_qa_report_deleted": True,
                "javascript_exceptions": 0,
                "cases": observations,
            }
            (out / "browser-checks.json").write_text(
                json.dumps(result, indent=2) + "\n"
            )
            print(json.dumps(result, indent=2))
            cmd("Browser.close")
    finally:
        if proc.poll() is None:
            proc.terminate()
            try:
                proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait(timeout=10)


if __name__ == "__main__":
    main()
