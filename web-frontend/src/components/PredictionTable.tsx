import type { PredictionResult } from "../types/prediction";

interface PredictionTableProps {
  data: PredictionResult[];
}

export default function PredictionTable({ data }: PredictionTableProps) {
  if (data.length === 0) return null;

  return (
    <table className="w-full border-collapse text-left text-gray-800">
      <thead className="bg-purple-200 text-purple-900">
        <tr>
          <th className="px-4 py-2 border-b border-gray-300">ID</th>
          <th className="px-4 py-2 border-b border-gray-300">Response</th>
        </tr>
      </thead>
      <tbody>
        {data.map((row, idx) => (
          <tr
            key={row.id}
            className={idx % 2 === 0 ? "bg-white/70" : "bg-purple-50"}
          >
            <td className="px-4 py-2 border-b border-gray-200">{row.id}</td>
            <td className="px-4 py-2 border-b border-gray-200">
              {row.response.toFixed(2)}
            </td>
          </tr>
        ))}
      </tbody>
    </table>
  );
}
