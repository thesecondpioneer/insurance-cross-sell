import * as Papa from "papaparse";
import { useState } from "react";
import type { PredictionResult } from "../types/prediction";
import PredictionTable from "./PredictionTable";

type CSVRow = {
  id: string;
  response: string;
};

export default function UploadCSV() {
  const [data, setData] = useState<PredictionResult[]>([]);
  const [predicted, setPredicted] = useState<PredictionResult[]>([]);
  const [error, setError] = useState<string | null>(null);

  const handleFile = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (!file) return;

    setError(null);
    setData([]);
    setPredicted([]);

    Papa.parse<CSVRow>(file, {
      header: true,
      skipEmptyLines: true,
      complete: (results) => {
        try {
          const rows = results.data;
          const parsed: PredictionResult[] = rows
            .filter(
              (row): row is CSVRow =>
                row.id !== undefined && row.response !== undefined
            )
            .map((row) => ({
              id: String(row.id),
              response: Number(row.response) || 0,
            }));
          setData(parsed);
        } catch (err: unknown) {
          setError(
            err instanceof Error
              ? `Ошибка при чтении CSV: ${err.message}`
              : "Ошибка при чтении CSV: неизвестная ошибка"
          );
        }
      },
      error: (err) => setError("Ошибка при чтении CSV: " + err.message),
    });
  };

  const handlePredict = () => {
    const predictedData = data.map((row) => ({
      ...row,
      response: Math.random(),
    }));
    setPredicted(predictedData);
  };

  const tableData = predicted.length > 0 ? predicted : data;

  return (
    <div className="max-w-3xl mx-auto p-6 upload-card">
      <h1 className="text-center text-3xl font-extrabold mb-6 text-purple-800">
        Insurance Predictions
      </h1>

      <p className="mb-2 text-gray-700">
        Загрузите CSV с колонками <code>id</code> и <code>response</code>. Пример:
      </p>

      <pre className="bg-purple-100 text-purple-800 p-3 mb-4 rounded-lg overflow-x-auto text-sm font-mono">
{`id,response
11504798,0
11504799,0
11504800,0`}
      </pre>

      {/* Buttons container */}
      <div className="mb-4 flex gap-4">
      {/* Browse CSV button */}
      <label className="flex-1">
        <span className="block w-full text-center bg-purple-500 text-white px-4 py-2 rounded cursor-pointer hover:bg-purple-600 transition">
          Browse CSV
        </span>
        <input
          type="file"
          accept=".csv"
          onChange={handleFile}
          className="hidden"
        />
      </label>

      {/* Predict button, same width as Browse */}
      {data.length > 0 && (
        <button
          className="flex-1 bg-purple-500 text-white px-4 py-2 rounded hover:bg-purple-600 transition"
          onClick={handlePredict}
        >
          Predict
        </button>
      )}
    </div>
      {error && <p className="text-red-500 mb-4">{error}</p>}

      {tableData.length > 0 && (
        <div className="max-h-80 overflow-y-auto rounded-lg border border-purple-200">
          <PredictionTable data={tableData} />
        </div>
      )}
    </div>
  );
}
