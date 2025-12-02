import type { PredictionResult } from "../types/prediction";

type Props = {
  data: PredictionResult[];
};

export default function PredictionTable({ data }: Props) {
  return (
    <table className="w-full border-collapse prediction-table">
      <thead>
        <tr>
          <th>id</th>
          <th>Gender</th>
          <th>Age</th>
          <th>Driving License</th>
          <th>Region Code</th>
          <th>Previously Insured</th>
          <th>Vehicle Age</th>
          <th>Vehicle Damage</th>
          <th>Annual Premium</th>
          <th>Policy Sales Channel</th>
          <th>Vintage</th>
          <th>Response</th>
        </tr>
      </thead>

      <tbody>
        {data.map((row, i) => (
          <tr key={i}>
            <td>{row.id}</td>
            <td>{row.Gender}</td>
            <td>{row.Age}</td>
            <td>{row.Driving_License}</td>
            <td>{row.Region_Code}</td>
            <td>{row.Previously_Insured}</td>
            <td>{row.Vehicle_Age}</td>
            <td>{row.Vehicle_Damage}</td>
            <td>{row.Annual_Premium}</td>
            <td>{row.Policy_Sales_Channel}</td>
            <td>{row.Vintage}</td>
            <td>{row.Response === -1 ? "" : row.Response}</td>
          </tr>
        ))}
      </tbody>
    </table>
  );
}
