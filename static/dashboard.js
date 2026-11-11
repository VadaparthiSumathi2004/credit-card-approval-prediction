async function loadDashboard(){
 const m=await fetch('/api/metrics'); if(!m.ok) throw new Error('Metrics not found. Run python train_model.py first.'); const data=await m.json();
 document.querySelector('#summary').innerHTML=`<div><b>${data.best_model}</b><span>Selected model</span></div><div><b>${data.test_rows}</b><span>Test rows</span></div><div><b>${data.dataset_rows}</b><span>Dataset rows</span></div><div><b>${data.selection_metric}</b><span>Selection method</span></div>`;
 const body=document.querySelector('#metricsTable tbody'); Object.entries(data.results).forEach(([name,v])=>{const tr=document.createElement('tr'); [name,v.accuracy,v.precision,v.recall,v.f1,v.roc_auc].forEach(x=>{const td=document.createElement('td');td.textContent=typeof x==='number'?(x*100).toFixed(1)+'%':x;tr.appendChild(td)});body.appendChild(tr)});
 const h=await fetch('/api/history'); const rows=await h.json(); const hb=document.querySelector('#historyTable tbody'); rows.forEach(r=>{const tr=document.createElement('tr');[r.id,r.created_at,r.label,r.probability+'%',r.model].forEach(v=>{const td=document.createElement('td');td.textContent=v;tr.appendChild(td)});hb.appendChild(tr)}); if(!rows.length) hb.innerHTML='<tr><td colspan="5">No predictions yet.</td></tr>';
}
loadDashboard().catch(e=>{document.querySelector('#summary').textContent=e.message});
