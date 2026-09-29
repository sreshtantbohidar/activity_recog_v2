"""Render a captured API payload with the dashboard's own analytics code.

The harness is the real dashboard plus a small injected script: it feeds a saved
payload to ``showPayload`` exactly as a finished job would, then reports how the
DOM came out (errors, KPI cards, tables, chips, missing fields) in a compact
``#__report`` element so the result can be read without dumping base64 images.
"""
import json

import sys
from pathlib import Path

DASHBOARD = Path("new_dashboard.html")
SERVED_PLACEHOLDER = "{{ max_upload_mb }}"

HEAD_PROBE = """<script>
window.__ERRORS__=[];
window.addEventListener('error',e=>window.__ERRORS__.push('error: '+(e.message||String(e.error))));
window.addEventListener('unhandledrejection',e=>window.__ERRORS__.push('rejection: '+String(e.reason&&e.reason.message||e.reason)));
</script>"""

TRIGGER = """<script>
(function(){
  var report={payload_keys:0,leaf_keys:0,missing:[],errors:window.__ERRORS__};
  function leafKeys(value,prefix,out){
    if(Array.isArray(value)){value.slice(0,3).forEach(function(item){leafKeys(item,prefix,out)});return out}
    if(value&&typeof value==='object'){Object.keys(value).forEach(function(key){leafKeys(value[key],key,out)});return out}
    if(prefix)out.push(prefix);
    return out;
  }
  try{
    report.payload_keys=Object.keys(window.__PAYLOAD__||{}).length;
    report.leaf_keys=leafKeys(window.__PAYLOAD__,null,[]).length;
    showPayload(window.__PAYLOAD__,window.__PAYLOAD__);
  }catch(error){report.errors.push('render threw: '+error.message)}
  var analytics=document.querySelector('#analytics');
  var text=analytics.textContent||'';
  function count(selector){return document.querySelectorAll(selector).length}
  report.hidden=analytics.hidden;
  report.kpis=Array.prototype.map.call(document.querySelectorAll('#analytics .kpi'),function(k){
    return (k.querySelector('.kpi-label')||{}).textContent+' = '+(k.querySelector('.kpi-value')||{}).textContent});
  report.cards=count('#analytics .viz-card');
  report.tables=count('#analytics table.viz-table');
  report.table_rows=count('#analytics table.viz-table tbody tr');
  report.dt_fields=count('#analytics .viz-fields dt');
  report.chips=count('#analytics .chip');
  report.minibars=count('#analytics .viz-mini');
  report.figures=count('#analytics .viz-figure');
  report.nested=count('#analytics .viz-nested');
  report.sections=Array.prototype.map.call(document.querySelectorAll('#analytics > .viz-object > .viz-card > .viz-title'),function(t){return t.textContent});
  // Every leaf key of the payload must be labelled somewhere on the page.
  var expected=[];
  leafKeys(window.__PAYLOAD__,null,[]).forEach(function(key){
    var label=humanKey(key);
    if(expected.indexOf(label)<0)expected.push(label);
  });
  expected.forEach(function(label){if(text.indexOf(label)<0)report.missing.push(label)});
  report.expected_labels=expected.length;
  report.raw_chars=(document.querySelector('#result')||{}).textContent.length;
  report.raw_starts=(document.querySelector('#result')||{}).textContent.slice(0,60);
  // The progress bar the upload flow drives while a job runs.
  setProgress(42,'annotating frame 27/781','running');
  report.progress={hidden:document.querySelector('#progressWrap').hidden,
    width:document.querySelector('#jobProgress').style.width,
    className:document.querySelector('#jobProgress').className,
    text:document.querySelector('#jobProgressText').textContent,
    stage:document.querySelector('#jobStage').textContent};
  hideProgress();
  report.progress_hidden_after=document.querySelector('#progressWrap').hidden;
  report.errors=window.__ERRORS__;
  report.analytics_chars=text.length;
  report.analytics_head=text.slice(0,1200);
  var box=document.createElement('div');
  box.id='__report';
  box.textContent=JSON.stringify(report);
  document.body.appendChild(box);
  document.title='RENDER-COMPLETE';
})();
</script>"""


def main():
    name = sys.argv[1]
    payload_path = Path(sys.argv[2])
    out_path = Path(sys.argv[3])

    payload = json.loads(payload_path.read_text(encoding="utf-8"))
    html = DASHBOARD.read_text(encoding="utf-8")
    if SERVED_PLACEHOLDER in html:
        html = html.replace(SERVED_PLACEHOLDER, "200")

    html = html.replace("<body>", "<body>" + HEAD_PROBE, 1)
    injected = ("<script>window.__PAYLOAD__="
                + json.dumps(payload, separators=(",", ":"))
                + ";</script>")
    # Feed the payload in before the dashboard script so it is defined by then,
    # then render with the dashboard's own functions once they exist.
    html = html.replace("<script>", injected + "<script>", 1)
    html = html.replace("</body>", TRIGGER + "</body>", 1)
    out_path.write_text(html, encoding="utf-8")
    print(f"wrote {out_path} for {name} ({payload_path.stat().st_size} bytes payload)")


if __name__ == "__main__":
    main()
