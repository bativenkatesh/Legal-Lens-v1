import React, { useState, useEffect } from "react";
import axios from "axios";

const API_BASE = "http://localhost:8000";
const TAX_SECTIONS = ["80D", "80G", "80E", "80GG", ""];

export default function OCRDashboard() {
  const [file, setFile] = useState(null);
  const [loading, setLoading] = useState(false);

  const [bills, setBills] = useState([]);

  const [financialYear, setFinancialYear] = useState("2017-18");
  const [summary, setSummary] = useState(null);

  /* ------------------ API CALLS ------------------ */

  const fetchBills = async () => {
    try {
      const res = await axios.get(`${API_BASE}/ocr/bills`);
      setBills(res.data);
    } catch (err) {
      console.error("Failed to fetch bills", err);
    }
  };

  const fetchSummary = async (fy) => {
    try {
      const res = await axios.get(`${API_BASE}/ocr/summary/${fy}`);
      setSummary(res.data);
    } catch (err) {
      console.error("Failed to fetch summary", err);
    }
  };

  useEffect(() => {
    fetchBills();
  }, []);

  useEffect(() => {
    fetchSummary(financialYear);
  }, [financialYear]);

  /* ------------------ ACTIONS ------------------ */

  const handleUpload = async () => {
    if (!file) return;

    const formData = new FormData();
    formData.append("file", file);

    setLoading(true);

    try {
      // 1️⃣ Upload file
      const uploadRes = await axios.post(
        `${API_BASE}/ocr/upload`,
        formData,
        {
          headers: {
            "Content-Type": "multipart/form-data",
          },
        }
      );

      const billId = uploadRes.data.bill_id;

      // 2️⃣ Auto-parse uploaded bill
      await axios.post(`${API_BASE}/ocr/parse/${billId}`);

      // 3️⃣ Refresh UI
      setFile(null);
      fetchBills();
      fetchSummary(financialYear);

    } catch (err) {
      console.error("Upload or parse failed", err);
    } finally {
      setLoading(false);
    }
  };

  const updateBill = async (bill_id, payload) => {
    try {
      await axios.patch(`${API_BASE}/ocr/bill/${bill_id}`, payload);
      fetchBills();
      fetchSummary(financialYear);
    } catch (err) {
      console.error("Failed to update bill", err);
    }
  };

  /* ------------------ UI ------------------ */

  return (
    <div className="p-6 max-w-7xl">
      <h1 className="text-2xl font-semibold mb-6">OCR Bills Dashboard</h1>

      {/* ---------------- Upload Section ---------------- */}
      <div className="border border-dashed border-gray-300 p-4 mb-8 rounded">
        <input
          type="file"
          accept=".pdf,.png,.jpg,.jpeg"
          onChange={(e) => setFile(e.target.files[0])}
        />

        <button
          onClick={handleUpload}
          disabled={!file || loading}
          className="ml-4 px-4 py-2 bg-blue-600 text-white rounded disabled:opacity-50"
        >
          {loading ? "Uploading..." : "Upload Bill"}
        </button>
      </div>

      {/* ---------------- Yearly Summary ---------------- */}
      <div className="mb-10 p-6 border rounded bg-gray-50">
        <div className="flex items-center justify-between mb-4">
          <h2 className="text-xl font-semibold">Yearly Tax Summary</h2>

          <select
            value={financialYear}
            onChange={(e) => setFinancialYear(e.target.value)}
            className="border rounded px-3 py-1"
          >
            <option value="2017-18">2017-18</option>
            <option value="2018-19">2018-19</option>
            <option value="2019-20">2019-20</option>
            <option value="2020-21">2020-21</option>
            <option value="2021-22">2021-22</option>
            <option value="2022-23">2022-23</option>
            <option value="2023-24">2023-24</option>
            <option value="2024-25">2024-25</option>
          </select>
        </div>

        {summary ? (
          <>
            <div className="grid grid-cols-2 gap-6 mb-6">
              <div className="p-4 bg-white rounded border">
                <p className="text-gray-500 text-sm">Total Bills</p>
                <p className="text-2xl font-semibold">
                  {summary.total_bills}
                </p>
              </div>

              <div className="p-4 bg-white rounded border">
                <p className="text-gray-500 text-sm">
                  Total Deduction Allowed
                </p>
                <p className="text-2xl font-semibold text-green-600">
                  ₹ {summary.total_deduction_allowed}
                </p>
              </div>
            </div>

            <h3 className="font-semibold mb-2">Section-wise Breakdown</h3>

            {Object.keys(summary.section_wise).length === 0 ? (
              <p className="text-gray-500">
                No tax-eligible bills for this year.
              </p>
            ) : (
              <table className="w-full border text-sm bg-white">
                <thead className="bg-gray-100">
                  <tr>
                    <th className="border p-2">Section</th>
                    <th className="border p-2">Claimed</th>
                    <th className="border p-2">Allowed</th>
                    <th className="border p-2">Limit</th>
                  </tr>
                </thead>
                <tbody>
                  {Object.entries(summary.section_wise).map(
                    ([section, data]) => (
                      <tr key={section}>
                        <td className="border p-2 text-center">{section}</td>
                        <td className="border p-2 text-center">
                          ₹ {data.claimed}
                        </td>
                        <td className="border p-2 text-center text-green-600">
                          ₹ {data.allowed}
                        </td>
                        <td className="border p-2 text-center">
                          {data.limit ? `₹ ${data.limit}` : "—"}
                        </td>
                      </tr>
                    )
                  )}
                </tbody>
              </table>
            )}
          </>
        ) : (
          <p className="text-gray-500">Loading summary...</p>
        )}
      </div>

      {/* ---------------- Bills Table ---------------- */}
      <h2 className="text-xl font-semibold mb-4">Uploaded Bills</h2>

      <table className="w-full border text-sm">
        <thead className="bg-gray-100">
          <tr>
            <th className="border p-2">Vendor</th>
            <th className="border p-2">Amount</th>
            <th className="border p-2">FY</th>
            <th className="border p-2">Eligible</th>
            <th className="border p-2">Section</th>
          </tr>
        </thead>
        <tbody>
          {bills.map((bill) => (
            <tr key={bill.bill_id}>
              <td className="border p-2">{bill.vendor || "-"}</td>
              <td className="border p-2 text-center">
                {bill.currency} {bill.total_amount}
              </td>
              <td className="border p-2 text-center">
                {bill.financial_year}
              </td>

              <td className="border p-2 text-center">
                <input
                  type="checkbox"
                  checked={bill.tax_eligible}
                  onChange={(e) =>
                    updateBill(bill.bill_id, {
                      tax_eligible: e.target.checked,
                      tax_section: e.target.checked
                        ? bill.tax_section || "80D"
                        : null,
                    })
                  }
                />
              </td>

              <td className="border p-2 text-center">
                <select
                  disabled={!bill.tax_eligible}
                  value={bill.tax_section || ""}
                  onChange={(e) =>
                    updateBill(bill.bill_id, {
                      tax_section: e.target.value,
                    })
                  }
                  className="border rounded px-2 py-1"
                >
                  <option value="">—</option>
                  {TAX_SECTIONS.filter(Boolean).map((sec) => (
                    <option key={sec} value={sec}>
                      {sec}
                    </option>
                  ))}
                </select>
              </td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}
