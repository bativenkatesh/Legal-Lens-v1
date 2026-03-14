import React, { useState, useEffect } from "react";
import { Link } from "react-router-dom";
import axios from "axios";
import {
  Upload,
  FileText,
  ArrowLeft,
  Receipt,
  TrendingUp,
  ChevronDown,
  CheckCircle2,
  Loader2,
  Sparkles,
  IndianRupee,
  BarChart3,
  CloudUpload,
} from "lucide-react";

const API_BASE = "http://localhost:8000";
const TAX_SECTIONS = ["80D", "80G", "80E", "80GG", ""];

export default function OCRDashboard() {
  const [file, setFile] = useState(null);
  const [loading, setLoading] = useState(false);
  const [bills, setBills] = useState([]);
  const [financialYear, setFinancialYear] = useState("2017-18");
  const [summary, setSummary] = useState(null);
  const [dragActive, setDragActive] = useState(false);

  /* API CALLS */
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

  /* ACTIONS */
  const handleUpload = async () => {
    if (!file) return;
    const formData = new FormData();
    formData.append("file", file);
    setLoading(true);
    try {
      const uploadRes = await axios.post(`${API_BASE}/ocr/upload`, formData, {
        headers: { "Content-Type": "multipart/form-data" },
      });
      const billId = uploadRes.data.bill_id;
      await axios.post(`${API_BASE}/ocr/parse/${billId}`);
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

  const handleDrag = (e) => {
    e.preventDefault();
    e.stopPropagation();
    if (e.type === "dragenter" || e.type === "dragover") {
      setDragActive(true);
    } else if (e.type === "dragleave") {
      setDragActive(false);
    }
  };

  const handleDrop = (e) => {
    e.preventDefault();
    e.stopPropagation();
    setDragActive(false);
    if (e.dataTransfer.files && e.dataTransfer.files[0]) {
      setFile(e.dataTransfer.files[0]);
    }
  };

  /* UI */
  return (
    <div className="min-h-screen bg-gray-50/50">
      {/* Header */}
      <header className="sticky top-0 z-10 bg-white border-b border-gray-100">
        <div className="max-w-7xl mx-auto px-6 py-4 flex items-center justify-between">
          <div className="flex items-center gap-4">
            <Link
              to="/chat"
              className="w-9 h-9 rounded-xl bg-gray-100 hover:bg-gray-200 flex items-center justify-center transition-colors"
            >
              <ArrowLeft className="w-4 h-4 text-gray-600" />
            </Link>
            <div className="flex items-center gap-2.5">
              <div className="w-9 h-9 rounded-xl bg-gradient-to-br from-blue-500 to-indigo-600 flex items-center justify-center shadow-sm shadow-blue-500/20">
                <Receipt className="w-4 h-4 text-white" />
              </div>
              <div>
                <h1 className="text-lg font-bold text-gray-900 leading-tight">
                  Bill Scanner
                </h1>
                <p className="text-xs text-gray-400 font-medium">
                  OCR-powered tax deduction tracker
                </p>
              </div>
            </div>
          </div>
          <Link
            to="/"
            className="flex items-center gap-2 text-sm font-semibold text-gray-500 hover:text-gray-800 transition-colors"
          >
            <Sparkles className="w-4 h-4" />
            Legal Lens
          </Link>
        </div>
      </header>

      <div className="max-w-7xl mx-auto px-6 py-8">
        {/* Upload Section */}
        <div
          className={`relative mb-10 rounded-2xl border-2 border-dashed transition-all p-8 ${dragActive
              ? "border-blue-400 bg-blue-50/50"
              : file
                ? "border-green-300 bg-green-50/30"
                : "border-gray-200 bg-white hover:border-gray-300"
            }`}
          onDragEnter={handleDrag}
          onDragLeave={handleDrag}
          onDragOver={handleDrag}
          onDrop={handleDrop}
        >
          <div className="flex flex-col md:flex-row items-center gap-6">
            <div className="flex-shrink-0">
              <div
                className={`w-16 h-16 rounded-2xl flex items-center justify-center ${file
                    ? "bg-green-100 text-green-600"
                    : "bg-blue-50 text-blue-500"
                  }`}
              >
                {file ? (
                  <CheckCircle2 className="w-7 h-7" />
                ) : (
                  <CloudUpload className="w-7 h-7" />
                )}
              </div>
            </div>

            <div className="flex-1 text-center md:text-left">
              {file ? (
                <>
                  <p className="text-base font-semibold text-gray-900 mb-1">
                    {file.name}
                  </p>
                  <p className="text-sm text-gray-400">
                    {(file.size / 1024).toFixed(1)} KB — Ready to upload
                  </p>
                </>
              ) : (
                <>
                  <p className="text-base font-semibold text-gray-900 mb-1">
                    Drop your bill here, or browse
                  </p>
                  <p className="text-sm text-gray-400">
                    Supports PDF, PNG, JPG files
                  </p>
                </>
              )}
            </div>

            <div className="flex items-center gap-3 flex-shrink-0">
              <label className="cursor-pointer px-5 py-2.5 rounded-xl bg-gray-100 text-gray-700 text-sm font-semibold hover:bg-gray-200 transition-all">
                Browse Files
                <input
                  type="file"
                  accept=".pdf,.png,.jpg,.jpeg"
                  onChange={(e) => setFile(e.target.files[0])}
                  className="hidden"
                />
              </label>

              <button
                onClick={handleUpload}
                disabled={!file || loading}
                className="px-6 py-2.5 rounded-xl bg-gradient-to-r from-blue-600 to-indigo-600 text-white text-sm font-bold disabled:opacity-40 disabled:cursor-not-allowed hover:shadow-lg hover:shadow-blue-500/25 hover:-translate-y-0.5 transition-all flex items-center gap-2"
              >
                {loading ? (
                  <>
                    <Loader2 className="w-4 h-4 animate-spin" />
                    Processing…
                  </>
                ) : (
                  <>
                    <Upload className="w-4 h-4" />
                    Upload Bill
                  </>
                )}
              </button>
            </div>
          </div>
        </div>

        {/* Yearly Summary */}
        <div className="mb-10">
          <div className="flex items-center justify-between mb-6">
            <div className="flex items-center gap-3">
              <div className="w-10 h-10 rounded-xl bg-indigo-50 flex items-center justify-center">
                <BarChart3 className="w-5 h-5 text-indigo-600" />
              </div>
              <div>
                <h2 className="text-xl font-bold text-gray-900">
                  Tax Summary
                </h2>
                <p className="text-xs text-gray-400 font-medium">
                  Financial year breakdown
                </p>
              </div>
            </div>

            <div className="relative">
              <select
                value={financialYear}
                onChange={(e) => setFinancialYear(e.target.value)}
                className="appearance-none bg-white border border-gray-200 rounded-xl px-4 py-2.5 pr-10 text-sm font-semibold text-gray-700 focus:outline-none focus:ring-2 focus:ring-blue-500/20 focus:border-blue-400 transition-all cursor-pointer hover:border-gray-300"
              >
                <option value="2017-18">FY 2017-18</option>
                <option value="2018-19">FY 2018-19</option>
                <option value="2019-20">FY 2019-20</option>
                <option value="2020-21">FY 2020-21</option>
                <option value="2021-22">FY 2021-22</option>
                <option value="2022-23">FY 2022-23</option>
                <option value="2023-24">FY 2023-24</option>
                <option value="2024-25">FY 2024-25</option>
              </select>
              <ChevronDown className="pointer-events-none absolute right-3 top-1/2 -translate-y-1/2 w-4 h-4 text-gray-400" />
            </div>
          </div>

          {summary ? (
            <>
              {/* Stat Cards */}
              <div className="grid grid-cols-1 sm:grid-cols-2 gap-4 mb-8">
                <div className="bg-white rounded-2xl border border-gray-100 p-6 hover:shadow-md hover:shadow-gray-100 transition-all">
                  <div className="flex items-center gap-3 mb-3">
                    <div className="w-10 h-10 rounded-xl bg-blue-50 flex items-center justify-center">
                      <FileText className="w-5 h-5 text-blue-600" />
                    </div>
                    <span className="text-sm font-medium text-gray-400">
                      Total Bills
                    </span>
                  </div>
                  <p className="text-3xl font-extrabold text-gray-900 tracking-tight">
                    {summary.total_bills}
                  </p>
                </div>

                <div className="bg-white rounded-2xl border border-gray-100 p-6 hover:shadow-md hover:shadow-gray-100 transition-all">
                  <div className="flex items-center gap-3 mb-3">
                    <div className="w-10 h-10 rounded-xl bg-green-50 flex items-center justify-center">
                      <TrendingUp className="w-5 h-5 text-green-600" />
                    </div>
                    <span className="text-sm font-medium text-gray-400">
                      Total Deduction Allowed
                    </span>
                  </div>
                  <p className="text-3xl font-extrabold text-green-600 tracking-tight">
                    ₹ {summary.total_deduction_allowed.toLocaleString()}
                  </p>
                </div>
              </div>

              {/* Section Breakdown */}
              <div className="bg-white rounded-2xl border border-gray-100 overflow-hidden">
                <div className="px-6 py-4 border-b border-gray-50">
                  <h3 className="text-sm font-bold text-gray-900 uppercase tracking-wider">
                    Section-wise Breakdown
                  </h3>
                </div>

                {Object.keys(summary.section_wise).length === 0 ? (
                  <div className="px-6 py-12 text-center">
                    <div className="w-12 h-12 rounded-full bg-gray-100 flex items-center justify-center mx-auto mb-3">
                      <IndianRupee className="w-5 h-5 text-gray-400" />
                    </div>
                    <p className="text-sm text-gray-400 font-medium">
                      No tax-eligible bills for this year
                    </p>
                  </div>
                ) : (
                  <table className="w-full">
                    <thead>
                      <tr className="bg-gray-50/50">
                        <th className="px-6 py-3.5 text-left text-xs font-bold text-gray-500 uppercase tracking-wider">
                          Section
                        </th>
                        <th className="px-6 py-3.5 text-right text-xs font-bold text-gray-500 uppercase tracking-wider">
                          Claimed
                        </th>
                        <th className="px-6 py-3.5 text-right text-xs font-bold text-gray-500 uppercase tracking-wider">
                          Allowed
                        </th>
                        <th className="px-6 py-3.5 text-right text-xs font-bold text-gray-500 uppercase tracking-wider">
                          Limit
                        </th>
                      </tr>
                    </thead>
                    <tbody className="divide-y divide-gray-50">
                      {Object.entries(summary.section_wise).map(
                        ([section, data]) => (
                          <tr
                            key={section}
                            className="hover:bg-gray-50/50 transition-colors"
                          >
                            <td className="px-6 py-4">
                              <span className="inline-flex items-center px-3 py-1 rounded-lg bg-indigo-50 text-indigo-700 text-sm font-bold">
                                {section}
                              </span>
                            </td>
                            <td className="px-6 py-4 text-right text-sm font-semibold text-gray-700">
                              ₹ {data.claimed.toLocaleString()}
                            </td>
                            <td className="px-6 py-4 text-right">
                              <span className="text-sm font-bold text-green-600">
                                ₹ {data.allowed.toLocaleString()}
                              </span>
                            </td>
                            <td className="px-6 py-4 text-right text-sm font-semibold text-gray-400">
                              {data.limit
                                ? `₹ ${data.limit.toLocaleString()}`
                                : "—"}
                            </td>
                          </tr>
                        )
                      )}
                    </tbody>
                  </table>
                )}
              </div>
            </>
          ) : (
            <div className="bg-white rounded-2xl border border-gray-100 p-12 text-center">
              <Loader2 className="w-6 h-6 animate-spin text-blue-500 mx-auto mb-3" />
              <p className="text-sm text-gray-400 font-medium">
                Loading summary…
              </p>
            </div>
          )}
        </div>

        {/* Bills Table */}
        <div className="bg-white rounded-2xl border border-gray-100 overflow-hidden">
          <div className="px-6 py-5 border-b border-gray-50 flex items-center justify-between">
            <div className="flex items-center gap-3">
              <div className="w-10 h-10 rounded-xl bg-amber-50 flex items-center justify-center">
                <Receipt className="w-5 h-5 text-amber-600" />
              </div>
              <div>
                <h2 className="text-base font-bold text-gray-900">
                  Uploaded Bills
                </h2>
                <p className="text-xs text-gray-400">
                  {bills.length} bill{bills.length !== 1 ? "s" : ""} processed
                </p>
              </div>
            </div>
          </div>

          {bills.length === 0 ? (
            <div className="px-6 py-16 text-center">
              <div className="w-14 h-14 rounded-2xl bg-gray-100 flex items-center justify-center mx-auto mb-4">
                <FileText className="w-6 h-6 text-gray-400" />
              </div>
              <p className="text-sm font-semibold text-gray-500 mb-1">
                No bills uploaded yet
              </p>
              <p className="text-xs text-gray-400">
                Upload a bill above to get started
              </p>
            </div>
          ) : (
            <div className="overflow-x-auto">
              <table className="w-full">
                <thead>
                  <tr className="bg-gray-50/50">
                    <th className="px-6 py-3.5 text-left text-xs font-bold text-gray-500 uppercase tracking-wider">
                      Vendor
                    </th>
                    <th className="px-6 py-3.5 text-right text-xs font-bold text-gray-500 uppercase tracking-wider">
                      Amount
                    </th>
                    <th className="px-6 py-3.5 text-center text-xs font-bold text-gray-500 uppercase tracking-wider">
                      FY
                    </th>
                    <th className="px-6 py-3.5 text-center text-xs font-bold text-gray-500 uppercase tracking-wider">
                      Eligible
                    </th>
                    <th className="px-6 py-3.5 text-center text-xs font-bold text-gray-500 uppercase tracking-wider">
                      Section
                    </th>
                  </tr>
                </thead>
                <tbody className="divide-y divide-gray-50">
                  {bills.map((bill) => (
                    <tr
                      key={bill.bill_id}
                      className="hover:bg-gray-50/50 transition-colors"
                    >
                      <td className="px-6 py-4">
                        <div className="flex items-center gap-3">
                          <div className="w-9 h-9 rounded-lg bg-gray-100 flex items-center justify-center flex-shrink-0">
                            <FileText className="w-4 h-4 text-gray-500" />
                          </div>
                          <span className="text-sm font-semibold text-gray-800">
                            {bill.vendor || "Unknown Vendor"}
                          </span>
                        </div>
                      </td>
                      <td className="px-6 py-4 text-right">
                        <span className="text-sm font-bold text-gray-900">
                          {bill.currency} {bill.total_amount?.toLocaleString()}
                        </span>
                      </td>
                      <td className="px-6 py-4 text-center">
                        <span className="inline-flex px-2.5 py-1 rounded-lg bg-gray-100 text-xs font-semibold text-gray-600">
                          {bill.financial_year}
                        </span>
                      </td>
                      <td className="px-6 py-4 text-center">
                        <label className="relative inline-flex items-center cursor-pointer">
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
                            className="sr-only peer"
                          />
                          <div className="w-9 h-5 bg-gray-200 peer-focus:ring-2 peer-focus:ring-blue-500/20 rounded-full peer peer-checked:after:translate-x-full peer-checked:after:border-white after:content-[''] after:absolute after:top-[2px] after:left-[2px] after:bg-white after:border-gray-300 after:border after:rounded-full after:h-4 after:w-4 after:transition-all peer-checked:bg-blue-600 after:shadow-sm"></div>
                        </label>
                      </td>
                      <td className="px-6 py-4 text-center">
                        <div className="relative inline-block">
                          <select
                            disabled={!bill.tax_eligible}
                            value={bill.tax_section || ""}
                            onChange={(e) =>
                              updateBill(bill.bill_id, {
                                tax_section: e.target.value,
                              })
                            }
                            className="appearance-none bg-white border border-gray-200 rounded-lg px-3 py-1.5 pr-8 text-sm font-semibold text-gray-700 focus:outline-none focus:ring-2 focus:ring-blue-500/20 focus:border-blue-400 disabled:opacity-40 disabled:cursor-not-allowed transition-all cursor-pointer"
                          >
                            <option value="">—</option>
                            {TAX_SECTIONS.filter(Boolean).map((sec) => (
                              <option key={sec} value={sec}>
                                {sec}
                              </option>
                            ))}
                          </select>
                          <ChevronDown className="pointer-events-none absolute right-2 top-1/2 -translate-y-1/2 w-3.5 h-3.5 text-gray-400" />
                        </div>
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
