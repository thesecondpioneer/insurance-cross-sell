export const API_URL = "http://localhost:8000";

export async function predictCSV(file: File) {
  const formData = new FormData();
  formData.append("file", file);

  const res = await fetch(`${API_URL}/predict-csv`, {
    method: "POST",
    body: formData,
  });

  if (!res.ok) {
    throw new Error("Prediction failed");
  }

  return res.json();
}
