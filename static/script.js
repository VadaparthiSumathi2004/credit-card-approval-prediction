const form = document.querySelector('#predictionForm');
const button = document.querySelector('#submitBtn');
const result = document.querySelector('#result');

form.addEventListener('submit', async (event) => {
  event.preventDefault();
  const payload = Object.fromEntries(new FormData(form).entries());
  for (const key of ['Income', 'Debt', 'YearsEmployed', 'CreditScore']) payload[key] = Number(payload[key]);
  for (const key of ['BankCustomer', 'PriorDefault', 'Employed']) payload[key] = Number(payload[key]);
  button.disabled = true;
  button.textContent = 'Analysing application…';
  result.hidden = true;
  result.className = 'result';
  try {
    const response = await fetch('/predict', {
      method: 'POST', headers: {'Content-Type': 'application/json'}, body: JSON.stringify(payload)
    });
    const data = await response.json();
    if (!response.ok) throw new Error(data.error || 'Unable to make prediction.');
    result.classList.add(data.approved ? 'positive' : 'caution');
    result.innerHTML = `<strong>${data.label}</strong><span>Estimated approval probability: ${data.approval_probability}%</span><small>Model: ${data.model}. ${data.disclaimer}</small>`;
  } catch (error) {
    result.classList.add('caution');
    result.textContent = error.message;
  } finally {
    result.hidden = false;
    button.disabled = false;
    button.innerHTML = 'Check eligibility <span>→</span>';
  }
});
const batchForm=document.querySelector('#batchForm');
if(batchForm){batchForm.addEventListener('submit',async(e)=>{e.preventDefault();const msg=document.querySelector('#batchMessage');msg.textContent='Processing CSV…';try{const res=await fetch('/batch-predict',{method:'POST',body:new FormData(batchForm)});if(!res.ok){const err=await res.json();throw new Error(err.error||'Batch request failed')}const blob=await res.blob();const url=URL.createObjectURL(blob);const a=document.createElement('a');a.href=url;a.download='credit_predictions.csv';a.click();URL.revokeObjectURL(url);msg.textContent='Completed. Your results CSV has been downloaded.'}catch(err){msg.textContent=err.message}})}
