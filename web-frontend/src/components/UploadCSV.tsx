import * as Papa from "papaparse";
import { useState } from "react";
import type { PredictionResult } from "../types/prediction";
import PredictionTable from "./PredictionTable";

type CSVRow = PredictionResult;

const MAX_PREVIEW_ROWS = 10000; // limit to avoid browser crash

export default function UploadCSV() {
  const [data, setData] = useState<PredictionResult[]>([]);
  const [predicted, setPredicted] = useState<PredictionResult[]>([]);
  const [error, setError] = useState<string | null>(null);

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
    const file = e.target.files?.[0];
    if (!file) return;

    setError(null);
    setData([]);
    setPredicted([]);

    const previewRows: PredictionResult[] = [];
    let headersValidated = false;
    let hasResponse = false;

    Papa.parse<CSVRow>(file, {
      header: true,
      skipEmptyLines: true,
      worker: true, // parse in a background thread
      step: (results, parser) => {
        const row = results.data;

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

        if (previewRows.length < MAX_PREVIEW_ROWS) {
          previewRows.push(parsedRow);
          setData([...previewRows]); // update preview as we go
        }
      },
      complete: () => {
        if (previewRows.length >= MAX_PREVIEW_ROWS) {
          setError(
            `Preview limited to ${MAX_PREVIEW_ROWS} rows. Full file can still be sent to backend.`
          );
        }
      },
      error: (err) => setError("Ошибка при чтении CSV: " + err.message),
    });
  };

  const handlePredict = () => {
    const predictedData = data.map((row) => ({
      ...row,
      Response: Math.random() > 0.5 ? 1 : 0,
    }));
    setPredicted(predictedData);
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
    <div className="max-w-6xl mx-auto p-6 upload-card">
      <h1 className="text-center text-3xl font-extrabold mb-6 text-purple-800">
        Insurance Predictions
      </h1>

      <p className="mb-2 text-gray-700">Upload a CSV of format:</p>

      <div className="table-wrapper">
        <PredictionTable data={exampleData} />
      </div>

      <div className="flex gap-4 mb-4 justify-center">
        {/* Browse CSV */}
        <label>
          <span className="button-common">Browse CSV</span>
          <input type="file" accept=".csv" onChange={handleFile} className="hidden" />
        </label>

        {/* Predict button */}
        {data.length > 0 && (
          <button className="button-common" onClick={handlePredict}>
            Predict
          </button>
        )}
      </div>

      {error && <p className="text-red-500 mb-4">{error}</p>}

      {/* Table */}
      {data.length > 0 && (
        <div className="table-wrapper">
          <PredictionTable data={predicted.length > 0 ? predicted : data} />
        </div>
      )}
    </div>
  );
}
