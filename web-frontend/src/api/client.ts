export async function predictCSV(file: File) {
  const formData = new FormData();
  formData.append("file", file);

  const res = await fetch('/api/predict-csv', {
    method: "POST",
    body: formData,
  });

  if (!res.ok) {
    throw new Error("Prediction failed");
  }

  const result = await res.json();
  return result.predictions;
}
