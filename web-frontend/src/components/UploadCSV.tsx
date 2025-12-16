import * as Papa from "papaparse";
import { useState } from "react";
import type { PredictionResult } from "../types/prediction";
import PredictionTable from "./PredictionTable";
import { predictCSV } from "../api/client";

type CSVRow = PredictionResult;

const MAX_PREVIEW_ROWS = 1000; // limit to avoid browser crash
const MAX_API_ROWS = 10000; // backend limit

export default function UploadCSV() {
  const [data, setData] = useState<PredictionResult[]>([]);           // 1000 строк для UI
  const [allRows, setAllRows] = useState<PredictionResult[]>([]);     // до 10k строк для экспорта
  const [predicted, setPredicted] = useState<PredictionResult[]>([]);
  const [error, setError] = useState<string | null>(null);
  const [file, setFile] = useState<File | null>(null);

  const requiredHeaders = [
    "id",
    "Gender",
    "Age",
    "Driving_License",
    "Region_Code",
    "Previously_Insured",
    "Vehicle_Age",
    "Vehicle_Damage",
    "Annual_Premium",
    "Policy_Sales_Channel",
    "Vintage",
  ];

  const optionalHeaders = ["Response"];

  const handleFile = (e: React.ChangeEvent<HTMLInputElement>) => {
    const selectedFile = e.target.files?.[0];
    if (!selectedFile) return;

    setFile(selectedFile);
    setError(null);
    setData([]);
    setAllRows([]);
    setPredicted([]);

    const previewRows: PredictionResult[] = [];
    const allRowsBuffer: PredictionResult[] = [];
    let headersValidated = false;
    let hasResponse = false;
    let rowCount = 0;

    Papa.parse<CSVRow>(selectedFile, {
      header: true,
      skipEmptyLines: true,
      worker: true,
      step: (results, parser) => {
        const row = results.data;
        rowCount++;

        // validate headers once
        if (!headersValidated) {
          for (const h of requiredHeaders) {
            if (!(h in row)) {
              setError(`Missing column: ${h}`);
              parser.abort();
              return;
            }
          }
          hasResponse = optionalHeaders.every((h) => h in row);
          headersValidated = true;
        }

        // parse row with types
        const parsedRow: PredictionResult = {
          id: Number(row.id),
          Gender: String(row.Gender).trim(),
          Age: Number(row.Age),
          Driving_License: Number(row.Driving_License),
          Region_Code: Number(row.Region_Code),
          Previously_Insured: Number(row.Previously_Insured),
          Vehicle_Age: String(row.Vehicle_Age).trim(),
          Vehicle_Damage: String(row.Vehicle_Damage).trim(),
          Annual_Premium: Number(row.Annual_Premium),
          Policy_Sales_Channel: Number(row.Policy_Sales_Channel),
          Vintage: Number(row.Vintage),
          Response: hasResponse ? Number(row.Response) : -1,
        };

        // UI: первые 1000 строк
        if (rowCount < MAX_PREVIEW_ROWS) {
          previewRows.push(parsedRow);
          setData([...previewRows]);
        }

        // Сохраняем до 10к строк для экспорта
        if (rowCount < MAX_API_ROWS) {
          allRowsBuffer.push(parsedRow);
        }

        // Останавливаемся на 10000 строке
        if (rowCount >= MAX_API_ROWS) {
          parser.abort();
        }
      },
      complete: () => {
        setAllRows(allRowsBuffer);
        if (rowCount > MAX_PREVIEW_ROWS) {
          setError(`Preview limited to ${MAX_PREVIEW_ROWS} rows. API requests are limited to ${MAX_API_ROWS} rows.`);
        }
      },
      error: (err) => setError("Ошибка при чтении CSV: " + err.message),
    });
  };

  async function createLimitedFile(file: File, maxRows: number): Promise<File> {
    const reader = file.stream().getReader();
    const decoder = new TextDecoder("utf-8");
    let result = await reader.read();
    let buffer = "";

    const lines: string[] = [];

    while (!result.done) {
      buffer += decoder.decode(result.value, { stream: true });
      const parts = buffer.split("\n");
      buffer = parts.pop() || "";

      for (const line of parts) {
        if (line.trim() === "") continue;
        lines.push(line);
        if (lines.length > maxRows) {
          break;
        }
      }

      if (lines.length > maxRows) {
        break;
      }

      result = await reader.read();
    }

    if (buffer && lines.length <= maxRows) {
      lines.push(buffer);
    }

    const limitedCsv = lines.slice(0, maxRows + 1).join("\n");
    return new File(
      [new Blob([limitedCsv], { type: "text/csv" })],
      "limited_data.csv",
      { type: "text/csv" }
    );
  }

const handlePredict = async () => {
  if (!file) return;

  try {
    const limitedFile = await createLimitedFile(file, MAX_API_ROWS);
    const predictedData: PredictionResult[] = await predictCSV(limitedFile);

    // Мержим по индексу, а не по id
    const finalAllRows = allRows.map((row, i) => {
      const pred = predictedData[i];
      return pred ? { ...row, Response: pred.Response } : row;
    });

    const finalPreview = finalAllRows.slice(0, MAX_PREVIEW_ROWS);
    setPredicted(finalPreview);
    setError(null);

    // Экспорт
    const exportData = finalAllRows;
    const exportContent = [
      Object.keys(exportData[0]!).join(","),
      ...exportData.map((row) => Object.values(row).map((v) => `"${v}"`).join(",")),
    ].join("\n");

    const exportBlob = new Blob([exportContent], { type: "text/csv" });
    const url = URL.createObjectURL(exportBlob);
    const link = document.createElement("a");
    link.href = url;
    link.download = "insurance_predictions.csv";
    link.click();
    URL.revokeObjectURL(url);
  } catch (err: unknown) {
    if (err instanceof Error) setError("Prediction failed: " + err.message);
    else setError("Prediction failed: unknown error");
  }
};


  const exampleData: PredictionResult[] = [
    {
      id: 1,
      Gender: "Male",
      Age: 23,
      Driving_License: 1,
      Region_Code: 5,
      Previously_Insured: 0,
      Vehicle_Age: ">2 Years",
      Vehicle_Damage: "Yes",
      Annual_Premium: 35000,
      Policy_Sales_Channel: 26,
      Vintage: 223,
      Response: -1,
    },
  ];

  return (
    <div className="upload-card">
      <h1 className="text-center text-3xl font-extrabold mb-6 text-purple-800">
        Insurance Predictions
      </h1>

      <p className="mb-2 text-gray-700">Upload a CSV of format:</p>

      <div className="table-wrapper">
        <PredictionTable data={exampleData} />
      </div>

      <div className="flex gap-4 mb-4 justify-center">
        <label>
          <span className="button-common">Browse CSV</span>
          <input type="file" accept=".csv" onChange={handleFile} className="hidden" />
        </label>

        {data.length > 0 && (
          <button className="button-common" onClick={handlePredict}>
            Predict
          </button>
        )}
      </div>

      {error && <p className="text-red-500 mb-4">{error}</p>}

      {data.length > 0 && (
        <div className="table-wrapper">
          <PredictionTable
            data={predicted.length > 0 ? predicted : data}
          />
        </div>
      )}
    </div>
  );
}
