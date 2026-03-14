import React, { useState, useRef, useEffect } from 'react'
import { useNavigate, Link } from "react-router-dom";
import axios from 'axios'
import {
    Sparkles, Send, Paperclip, Image, File, PiggyBank, Calculator, TrendingUp,
    Settings, HelpCircle, LogOut, ChevronDown, MessageSquare, Plus, Trash2,
    Receipt, Upload, FileText, CheckCircle2, Loader2, IndianRupee, BarChart3,
    CloudUpload, ArrowLeft, Eye, X, AlertTriangle, FileSpreadsheet, Search, Globe, Library, Folder, History, Command
} from 'lucide-react'
import '../App.css'

const API_BASE_URL = 'http://localhost:8000'

/* ============================================================
   OCR Bill Scanner Panel (embedded)
   ============================================================ */
const TAX_SECTIONS = ["80D", "80G", "80E", "80GG", ""];

function BillScannerPanel() {
    const [file, setFile] = useState(null);
    const [loading, setLoading] = useState(false);
    const [bills, setBills] = useState([]);
    const [financialYear, setFinancialYear] = useState("all");
    const [summary, setSummary] = useState(null);
    const [dragActive, setDragActive] = useState(false);
    const [previewBillId, setPreviewBillId] = useState(null);
    const [billToDelete, setBillToDelete] = useState(null);

    // Close preview on Escape key
    useEffect(() => {
        const handleEsc = (e) => { if (e.key === 'Escape') setPreviewBillId(null); };
        if (previewBillId) window.addEventListener('keydown', handleEsc);
        return () => window.removeEventListener('keydown', handleEsc);
    }, [previewBillId]);

    const fetchBills = async () => {
        try { setBills((await axios.get(`${API_BASE_URL}/ocr/bills`)).data); }
        catch (err) { console.error("Failed to fetch bills", err); }
    };

    const fetchSummary = async (fy) => {
        try { setSummary((await axios.get(`${API_BASE_URL}/ocr/summary/${fy}`)).data); }
        catch (err) { console.error("Failed to fetch summary", err); }
    };

    useEffect(() => { fetchBills(); }, []);
    useEffect(() => { fetchSummary(financialYear); }, [financialYear]);

    const handleUpload = async () => {
        if (!file) return;
        const formData = new FormData();
        formData.append("file", file);
        setLoading(true);
        try {
            const uploadRes = await axios.post(`${API_BASE_URL}/ocr/upload`, formData, {
                headers: { "Content-Type": "multipart/form-data" },
            });
            await axios.post(`${API_BASE_URL}/ocr/parse/${uploadRes.data.bill_id}`);
            setFile(null);
            fetchBills();
            fetchSummary(financialYear);
        } catch (err) { console.error("Upload or parse failed", err); }
        finally { setLoading(false); }
    };

    const updateBill = async (bill_id, payload) => {
        try {
            await axios.patch(`${API_BASE_URL}/ocr/bill/${bill_id}`, payload);
            fetchBills();
            fetchSummary(financialYear);
        } catch (err) { console.error("Failed to update bill", err); }
    };

    const confirmDeleteBill = async () => {
        if (!billToDelete) return;
        try {
            await axios.delete(`${API_BASE_URL}/ocr/bill/${billToDelete}`);
            setBillToDelete(null);
            fetchBills();
            fetchSummary(financialYear);
        } catch (err) { console.error("Failed to delete bill", err); }
    };

    const handleDrag = (e) => {
        e.preventDefault(); e.stopPropagation();
        if (e.type === "dragenter" || e.type === "dragover") setDragActive(true);
        else if (e.type === "dragleave") setDragActive(false);
    };

    const handleDrop = (e) => {
        e.preventDefault(); e.stopPropagation(); setDragActive(false);
        if (e.dataTransfer.files?.[0]) setFile(e.dataTransfer.files[0]);
    };

    return (
        <div className="px-8 py-8">
            {/* Upload */}
            <div
                className={`relative mb-10 rounded-2xl border-2 border-dashed transition-all p-8 ${dragActive ? "border-blue-400 bg-blue-50/50"
                    : file ? "border-green-300 bg-green-50/30"
                        : "border-gray-200 bg-white hover:border-gray-300"
                    }`}
                onDragEnter={handleDrag} onDragLeave={handleDrag}
                onDragOver={handleDrag} onDrop={handleDrop}
            >
                <div className="flex flex-col md:flex-row items-center gap-6">
                    <div className={`w-16 h-16 rounded-2xl flex items-center justify-center flex-shrink-0 ${file ? "bg-green-100 text-green-600" : "bg-blue-50 text-blue-500"
                        }`}>
                        {file ? <CheckCircle2 className="w-7 h-7" /> : <CloudUpload className="w-7 h-7" />}
                    </div>
                    <div className="flex-1 text-center md:text-left">
                        {file ? (
                            <>
                                <p className="text-base font-semibold text-gray-900 mb-1">{file.name}</p>
                                <p className="text-sm text-gray-400">{(file.size / 1024).toFixed(1)} KB — Ready to upload</p>
                            </>
                        ) : (
                            <>
                                <p className="text-base font-semibold text-gray-900 mb-1">Drop your bill here, or browse</p>
                                <p className="text-sm text-gray-400">Supports PDF, PNG, JPG files</p>
                            </>
                        )}
                    </div>
                    <div className="flex items-center gap-3 flex-shrink-0">
                        <label className="cursor-pointer px-5 py-2.5 rounded-xl bg-gray-100 text-gray-700 text-sm font-semibold hover:bg-gray-200 transition-all">
                            Browse Files
                            <input type="file" accept=".pdf,.png,.jpg,.jpeg" onChange={(e) => setFile(e.target.files[0])} className="hidden" />
                        </label>
                        <button onClick={handleUpload} disabled={!file || loading}
                            className="px-6 py-2.5 rounded-xl bg-gradient-to-r from-blue-600 to-indigo-600 text-white text-sm font-bold disabled:opacity-40 disabled:cursor-not-allowed hover:shadow-lg hover:shadow-blue-500/25 hover:-translate-y-0.5 transition-all flex items-center gap-2">
                            {loading ? <><Loader2 className="w-4 h-4 animate-spin" />Processing…</> : <><Upload className="w-4 h-4" />Upload Bill</>}
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
                            <h2 className="text-xl font-bold text-gray-900">Tax Summary</h2>
                            <p className="text-xs text-gray-400 font-medium">Financial year breakdown</p>
                        </div>
                    </div>
                    <div className="relative">
                        <select value={financialYear} onChange={(e) => setFinancialYear(e.target.value)}
                            className="appearance-none bg-white border border-gray-200 rounded-xl px-4 py-2.5 pr-10 text-sm font-semibold text-gray-700 focus:outline-none focus:ring-2 focus:ring-blue-500/20 focus:border-blue-400 transition-all cursor-pointer hover:border-gray-300">
                            <option value="all">All Years</option>
                            {["2024-25", "2023-24", "2022-23", "2021-22", "2020-21", "2019-20", "2018-19", "2017-18"].map(y =>
                                <option key={y} value={y}>FY {y}</option>
                            )}
                        </select>
                        <ChevronDown className="pointer-events-none absolute right-3 top-1/2 -translate-y-1/2 w-4 h-4 text-gray-400" />
                    </div>
                </div>

                {summary ? (
                    <>
                        <div className="grid grid-cols-1 sm:grid-cols-2 gap-4 mb-8">
                            <div className="bg-white rounded-2xl border border-gray-100 p-6 hover:shadow-md hover:shadow-gray-100 transition-all">
                                <div className="flex items-center gap-3 mb-3">
                                    <div className="w-10 h-10 rounded-xl bg-blue-50 flex items-center justify-center"><FileText className="w-5 h-5 text-blue-600" /></div>
                                    <span className="text-sm font-medium text-gray-400">Total Bills</span>
                                </div>
                                <p className="text-3xl font-extrabold text-gray-900 tracking-tight">{summary.total_bills}</p>
                            </div>
                            <div className="bg-white rounded-2xl border border-gray-100 p-6 hover:shadow-md hover:shadow-gray-100 transition-all">
                                <div className="flex items-center gap-3 mb-3">
                                    <div className="w-10 h-10 rounded-xl bg-green-50 flex items-center justify-center"><TrendingUp className="w-5 h-5 text-green-600" /></div>
                                    <span className="text-sm font-medium text-gray-400">Total Deduction Allowed</span>
                                </div>
                                <p className="text-3xl font-extrabold text-green-600 tracking-tight">₹ {summary.total_deduction_allowed?.toLocaleString()}</p>
                            </div>
                        </div>

                        {/* Breakdown table */}
                        <div className="bg-white rounded-2xl border border-gray-100 overflow-hidden">
                            <div className="px-6 py-4 border-b border-gray-50">
                                <h3 className="text-sm font-bold text-gray-900 uppercase tracking-wider">Section-wise Breakdown</h3>
                            </div>
                            {Object.keys(summary.section_wise).length === 0 ? (
                                <div className="px-6 py-12 text-center">
                                    <div className="w-12 h-12 rounded-full bg-gray-100 flex items-center justify-center mx-auto mb-3"><IndianRupee className="w-5 h-5 text-gray-400" /></div>
                                    <p className="text-sm text-gray-400 font-medium">No tax-eligible bills for this year</p>
                                </div>
                            ) : (
                                <table className="w-full">
                                    <thead><tr className="bg-gray-50/50">
                                        <th className="px-6 py-3.5 text-left text-xs font-bold text-gray-500 uppercase tracking-wider">Section</th>
                                        <th className="px-6 py-3.5 text-right text-xs font-bold text-gray-500 uppercase tracking-wider">Claimed</th>
                                        <th className="px-6 py-3.5 text-right text-xs font-bold text-gray-500 uppercase tracking-wider">Allowed</th>
                                        <th className="px-6 py-3.5 text-right text-xs font-bold text-gray-500 uppercase tracking-wider">Limit</th>
                                    </tr></thead>
                                    <tbody className="divide-y divide-gray-50">
                                        {Object.entries(summary.section_wise).map(([section, data]) => (
                                            <tr key={section} className="hover:bg-gray-50/50 transition-colors">
                                                <td className="px-6 py-4"><span className="inline-flex items-center px-3 py-1 rounded-lg bg-indigo-50 text-indigo-700 text-sm font-bold">{section}</span></td>
                                                <td className="px-6 py-4 text-right text-sm font-semibold text-gray-700">₹ {data.claimed?.toLocaleString()}</td>
                                                <td className="px-6 py-4 text-right"><span className="text-sm font-bold text-green-600">₹ {data.allowed?.toLocaleString()}</span></td>
                                                <td className="px-6 py-4 text-right text-sm font-semibold text-gray-400">{data.limit ? `₹ ${data.limit.toLocaleString()}` : "—"}</td>
                                            </tr>
                                        ))}
                                    </tbody>
                                </table>
                            )}
                        </div>
                    </>
                ) : (
                    <div className="bg-white rounded-2xl border border-gray-100 p-12 text-center">
                        <Loader2 className="w-6 h-6 animate-spin text-blue-500 mx-auto mb-3" />
                        <p className="text-sm text-gray-400 font-medium">Loading summary…</p>
                    </div>
                )}
            </div>

            {/* Bills Table */}
            <div className="bg-white rounded-2xl border border-gray-100 overflow-hidden">
                <div className="px-6 py-5 border-b border-gray-50 flex items-center gap-3">
                    <div className="w-10 h-10 rounded-xl bg-amber-50 flex items-center justify-center"><Receipt className="w-5 h-5 text-amber-600" /></div>
                    <div>
                        <h2 className="text-base font-bold text-gray-900">Uploaded Bills</h2>
                        <p className="text-xs text-gray-400">{bills.length} bill{bills.length !== 1 ? "s" : ""} processed</p>
                    </div>
                </div>
                {bills.length === 0 ? (
                    <div className="px-6 py-16 text-center">
                        <div className="w-14 h-14 rounded-2xl bg-gray-100 flex items-center justify-center mx-auto mb-4"><FileText className="w-6 h-6 text-gray-400" /></div>
                        <p className="text-sm font-semibold text-gray-500 mb-1">No bills uploaded yet</p>
                        <p className="text-xs text-gray-400">Upload a bill above to get started</p>
                    </div>
                ) : (
                    <div className="overflow-x-auto">
                        <table className="w-full">
                            <thead><tr className="bg-gray-50/50">
                                <th className="px-6 py-3.5 text-left text-xs font-bold text-gray-500 uppercase tracking-wider">Vendor</th>
                                <th className="px-6 py-3.5 text-right text-xs font-bold text-gray-500 uppercase tracking-wider">Amount</th>
                                <th className="px-6 py-3.5 text-center text-xs font-bold text-gray-500 uppercase tracking-wider">FY</th>
                                <th className="px-6 py-3.5 text-center text-xs font-bold text-gray-500 uppercase tracking-wider">Eligible</th>
                                <th className="px-6 py-3.5 text-center text-xs font-bold text-gray-500 uppercase tracking-wider">Section</th>
                                <th className="px-6 py-3.5 text-center text-xs font-bold text-gray-500 uppercase tracking-wider">View</th>
                            </tr></thead>
                            <tbody className="divide-y divide-gray-50">
                                {bills.map((bill) => (
                                    <tr key={bill.bill_id} className="hover:bg-gray-50/50 transition-colors">
                                        <td className="px-6 py-4">
                                            <div className="flex items-center gap-3">
                                                <div className="w-9 h-9 rounded-lg bg-gray-100 flex items-center justify-center flex-shrink-0"><FileText className="w-4 h-4 text-gray-500" /></div>
                                                <span className="text-sm font-semibold text-gray-800">{bill.vendor || "Unknown Vendor"}</span>
                                            </div>
                                        </td>
                                        <td className="px-6 py-4 text-right"><span className="text-sm font-bold text-gray-900">{bill.currency} {bill.total_amount?.toLocaleString()}</span></td>
                                        <td className="px-6 py-4 text-center"><span className="inline-flex px-2.5 py-1 rounded-lg bg-gray-100 text-xs font-semibold text-gray-600">{bill.financial_year}</span></td>
                                        <td className="px-6 py-4 text-center">
                                            <label className="relative inline-flex items-center cursor-pointer">
                                                <input type="checkbox" checked={bill.tax_eligible}
                                                    onChange={(e) => updateBill(bill.bill_id, { tax_eligible: e.target.checked, tax_section: e.target.checked ? bill.tax_section || "80D" : null })}
                                                    className="sr-only peer" />
                                                <div className="w-9 h-5 bg-gray-200 peer-focus:ring-2 peer-focus:ring-blue-500/20 rounded-full peer peer-checked:after:translate-x-full peer-checked:after:border-white after:content-[''] after:absolute after:top-[2px] after:left-[2px] after:bg-white after:border-gray-300 after:border after:rounded-full after:h-4 after:w-4 after:transition-all peer-checked:bg-blue-600 after:shadow-sm"></div>
                                            </label>
                                        </td>
                                        <td className="px-6 py-4 text-center">
                                            <div className="relative inline-block">
                                                <select disabled={!bill.tax_eligible} value={bill.tax_section || ""}
                                                    onChange={(e) => updateBill(bill.bill_id, { tax_section: e.target.value })}
                                                    className="appearance-none bg-white border border-gray-200 rounded-lg px-3 py-1.5 pr-8 text-sm font-semibold text-gray-700 focus:outline-none focus:ring-2 focus:ring-blue-500/20 focus:border-blue-400 disabled:opacity-40 disabled:cursor-not-allowed transition-all cursor-pointer">
                                                    <option value="">—</option>
                                                    {TAX_SECTIONS.filter(Boolean).map((sec) => <option key={sec} value={sec}>{sec}</option>)}
                                                </select>
                                                <ChevronDown className="pointer-events-none absolute right-2 top-1/2 -translate-y-1/2 w-3.5 h-3.5 text-gray-400" />
                                            </div>
                                        </td>
                                        <td className="px-6 py-4 text-center">
                                            <div className="flex items-center justify-center gap-2">
                                                <button
                                                    onClick={() => setPreviewBillId(bill.bill_id)}
                                                    className="inline-flex items-center gap-1.5 px-3 py-1.5 rounded-lg bg-blue-50 text-blue-600 text-xs font-semibold hover:bg-blue-100 transition-all"
                                                >
                                                    <Eye className="w-3.5 h-3.5" />
                                                    View
                                                </button>
                                                <button
                                                    onClick={() => setBillToDelete(bill.bill_id)}
                                                    className="inline-flex items-center gap-1.5 px-3 py-1.5 rounded-lg bg-red-50 text-red-500 text-xs font-semibold hover:bg-red-100 transition-all"
                                                >
                                                    <Trash2 className="w-3.5 h-3.5" />
                                                    Delete
                                                </button>
                                            </div>
                                        </td>
                                    </tr>
                                ))}
                            </tbody>
                        </table>
                    </div>
                )}
            </div>

            {/* Bill Preview Modal */}
            {previewBillId && (
                <div
                    className="fixed inset-0 z-50 flex items-center justify-center bg-black/50 backdrop-blur-sm"
                    onClick={() => setPreviewBillId(null)}
                >
                    <div
                        className="relative bg-white rounded-2xl shadow-2xl max-w-3xl w-full mx-4 max-h-[85vh] flex flex-col overflow-hidden"
                        onClick={(e) => e.stopPropagation()}
                    >
                        {/* Modal Header */}
                        <div className="flex items-center justify-between px-6 py-4 border-b border-gray-100">
                            <div className="flex items-center gap-3">
                                <div className="w-9 h-9 rounded-xl bg-blue-50 flex items-center justify-center">
                                    <FileText className="w-4 h-4 text-blue-600" />
                                </div>
                                <div>
                                    <h3 className="text-sm font-bold text-gray-900">Bill Preview</h3>
                                    <p className="text-xs text-gray-400 font-mono">{previewBillId.substring(0, 8)}…</p>
                                </div>
                            </div>
                            <button
                                onClick={() => setPreviewBillId(null)}
                                className="w-8 h-8 rounded-lg bg-gray-100 hover:bg-gray-200 flex items-center justify-center transition-colors"
                            >
                                <X className="w-4 h-4 text-gray-500" />
                            </button>
                        </div>
                        {/* Modal Body */}
                        <div className="flex-1 overflow-auto p-6 flex items-center justify-center bg-gray-50">
                            <img
                                src={`${API_BASE_URL}/ocr/file/${previewBillId}`}
                                alt="Bill preview"
                                className="max-w-full max-h-[65vh] object-contain rounded-lg shadow-sm"
                                onError={(e) => {
                                    // If it's a PDF, swap to an iframe
                                    const parent = e.target.parentNode;
                                    const iframe = document.createElement('iframe');
                                    iframe.src = `${API_BASE_URL}/ocr/file/${previewBillId}`;
                                    iframe.className = 'w-full h-[65vh] rounded-lg border border-gray-200';
                                    iframe.title = 'Bill PDF';
                                    parent.replaceChild(iframe, e.target);
                                }}
                            />
                        </div>
                    </div>
                </div>
            )}

            {/* Delete Confirmation Modal */}
            {billToDelete && (
                <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/50 backdrop-blur-sm">
                    <div className="relative bg-white rounded-2xl shadow-xl max-w-sm w-full mx-4 p-6" onClick={(e) => e.stopPropagation()}>
                        <div className="flex flex-col items-center text-center">
                            <div className="w-12 h-12 rounded-full bg-red-50 flex items-center justify-center mb-4">
                                <AlertTriangle className="w-6 h-6 text-red-500" />
                            </div>
                            <h3 className="text-lg font-bold text-gray-900 mb-2">Delete Bill</h3>
                            <p className="text-sm text-gray-500 mb-6">
                                Are you sure you want to delete this bill? This action cannot be undone.
                            </p>
                            <div className="flex gap-3 w-full">
                                <button
                                    onClick={() => setBillToDelete(null)}
                                    className="flex-1 px-4 py-2.5 rounded-xl bg-gray-100 text-gray-700 text-sm font-semibold hover:bg-gray-200 transition-all"
                                >
                                    Cancel
                                </button>
                                <button
                                    onClick={confirmDeleteBill}
                                    className="flex-1 px-4 py-2.5 rounded-xl bg-red-500 text-white text-sm font-semibold hover:bg-red-600 transition-all"
                                >
                                    Delete
                                </button>
                            </div>
                        </div>
                    </div>
                </div>
            )}
        </div>
    );
}

/* ============================================================
   GST Invoice Panel (embedded)
   ============================================================ */
function GSTInvoicePanel() {
    const [file, setFile] = useState(null);
    const [loading, setLoading] = useState(false);
    const [dragActive, setDragActive] = useState(false);
    const [lastExtracted, setLastExtracted] = useState(null);

    const handleDrag = (e) => {
        e.preventDefault(); e.stopPropagation();
        if (e.type === "dragenter" || e.type === "dragover") setDragActive(true);
        else if (e.type === "dragleave") setDragActive(false);
    };

    const handleDrop = (e) => {
        e.preventDefault(); e.stopPropagation(); setDragActive(false);
        if (e.dataTransfer.files?.[0]) setFile(e.dataTransfer.files[0]);
    };

    const handleUpload = async () => {
        if (!file) return;
        const formData = new FormData();
        formData.append("file", file);
        setLoading(true);
        try {
            const res = await axios.post(`${API_BASE_URL}/ocr/gst/upload`, formData, {
                headers: { "Content-Type": "multipart/form-data" },
            });
            setLastExtracted(res.data.parsed_data);
            setFile(null);
            alert("Invoice successfully parsed and appended to Excel!");
        } catch (err) {
            console.error("GST Upload failed", err);
            alert("Failed to process GST invoice.");
        } finally {
            setLoading(false);
        }
    };

    const downloadExcel = () => {
        window.open(`${API_BASE_URL}/ocr/gst/excel`, "_blank");
    };

    const handleResetExcel = async () => {
        if (!window.confirm("Are you sure you want to reset the Excel file? All appended data will be lost.")) return;
        try {
            await axios.delete(`${API_BASE_URL}/ocr/gst/excel`);
            setLastExtracted(null);
            alert("Excel file has been reset.");
        } catch (err) {
            console.error("Reset failed", err);
            alert("Failed to reset Excel file.");
        }
    };

    return (
        <div className="px-8 py-8">
            <div className="flex items-center justify-between mb-8">
                <div>
                    <h2 className="text-2xl font-bold text-gray-900">GST Invoice Scanner</h2>
                    <p className="text-sm text-gray-500 mt-1">Extract specific GST fields directly into a master Excel file.</p>
                </div>
                <div className="flex items-center gap-3">
                    <button
                        onClick={handleResetExcel}
                        className="flex items-center gap-2 px-5 py-2.5 rounded-xl bg-red-50 text-red-700 text-sm font-bold border border-red-100 hover:bg-red-100 transition-all"
                    >
                        <Trash2 className="w-4 h-4 text-red-500" />
                        Reset Data
                    </button>
                    <button
                        onClick={downloadExcel}
                        className="flex items-center gap-2 px-5 py-2.5 rounded-xl bg-green-50 text-green-700 text-sm font-bold border border-green-200/60 hover:bg-green-100 transition-all"
                    >
                        <FileSpreadsheet className="w-5 h-5 text-green-600" />
                        Download Excel
                    </button>
                </div>
            </div>

            {/* Upload Area */}
            <div
                className={`relative mb-10 rounded-2xl border-2 border-dashed transition-all p-8 ${dragActive ? "border-indigo-400 bg-indigo-50/50"
                    : file ? "border-green-300 bg-green-50/30"
                        : "border-gray-200 bg-white hover:border-gray-300"
                    }`}
                onDragEnter={handleDrag} onDragLeave={handleDrag}
                onDragOver={handleDrag} onDrop={handleDrop}
            >
                <div className="flex flex-col md:flex-row items-center gap-6">
                    <div className={`w-16 h-16 rounded-2xl flex items-center justify-center flex-shrink-0 ${file ? "bg-green-100 text-green-600" : "bg-indigo-50 text-indigo-500"
                        }`}>
                        {file ? <CheckCircle2 className="w-7 h-7" /> : <CloudUpload className="w-7 h-7" />}
                    </div>
                    <div className="flex-1 text-center md:text-left">
                        {file ? (
                            <>
                                <p className="text-base font-semibold text-gray-900 mb-1">{file.name}</p>
                                <p className="text-sm text-gray-400">{(file.size / 1024).toFixed(1)} KB — Ready to upload</p>
                            </>
                        ) : (
                            <>
                                <p className="text-base font-semibold text-gray-900 mb-1">Drop your GST invoice here, or browse</p>
                                <p className="text-sm text-gray-400">Supports PDF, PNG, JPG files</p>
                            </>
                        )}
                    </div>
                    <div className="flex items-center gap-3 flex-shrink-0">
                        <label className="cursor-pointer px-5 py-2.5 rounded-xl bg-gray-100 text-gray-700 text-sm font-semibold hover:bg-gray-200 transition-all">
                            Browse Files
                            <input type="file" accept=".pdf,.png,.jpg,.jpeg" onChange={(e) => setFile(e.target.files[0])} className="hidden" />
                        </label>
                        <button onClick={handleUpload} disabled={!file || loading}
                            className="px-6 py-2.5 rounded-xl bg-gradient-to-r from-indigo-600 to-purple-600 text-white text-sm font-bold disabled:opacity-40 disabled:cursor-not-allowed hover:shadow-lg hover:shadow-indigo-500/25 hover:-translate-y-0.5 transition-all flex items-center gap-2">
                            {loading ? <><Loader2 className="w-4 h-4 animate-spin" />Processing…</> : <><Upload className="w-4 h-4" />Extract to Excel</>}
                        </button>
                    </div>
                </div>
            </div>

            {/* Last Extracted Data Preview */}
            {lastExtracted && (
                <div className="bg-white rounded-2xl border border-gray-100 overflow-hidden">
                    <div className="px-6 py-4 border-b border-gray-50 flex items-center justify-between">
                        <h3 className="text-sm font-bold text-gray-900 uppercase tracking-wider">Latest Extraction Result</h3>
                        <span className="text-xs font-semibold text-green-600 bg-green-50 px-2 py-1 rounded-md">Appended to Excel</span>
                    </div>
                    <div className="p-6 grid grid-cols-2 lg:grid-cols-4 gap-6">
                        <div>
                            <p className="text-xs text-gray-400 font-semibold mb-1 uppercase">Type</p>
                            <p className="text-sm font-bold text-gray-900">{lastExtracted.type || "—"}</p>
                        </div>
                        <div>
                            <p className="text-xs text-gray-400 font-semibold mb-1 uppercase">POS</p>
                            <p className="text-sm font-bold text-gray-900">{lastExtracted.place_of_supply || "—"}</p>
                        </div>
                        <div>
                            <p className="text-xs text-gray-400 font-semibold mb-1 uppercase">Applicable % Rate</p>
                            <p className="text-sm font-bold text-gray-900">{lastExtracted.applicable_tax_rate_percent || "—"}</p>
                        </div>
                        <div>
                            <p className="text-xs text-gray-400 font-semibold mb-1 uppercase">Rate</p>
                            <p className="text-sm font-bold text-gray-900">{lastExtracted.rate || "—"}</p>
                        </div>
                        <div>
                            <p className="text-xs text-gray-400 font-semibold mb-1 uppercase">Taxable Value</p>
                            <p className="text-sm font-bold text-gray-900">
                                ₹ {lastExtracted.taxable_value?.toLocaleString(undefined, { minimumFractionDigits: 2 })}
                            </p>
                        </div>
                        <div>
                            <p className="text-xs text-gray-400 font-semibold mb-1 uppercase">Total Amount</p>
                            <p className="text-sm font-bold text-indigo-600">
                                ₹ {lastExtracted.total_amount?.toLocaleString(undefined, { minimumFractionDigits: 2 })}
                            </p>
                        </div>
                        <div>
                            <p className="text-xs text-gray-400 font-semibold mb-1 uppercase">GSTIN</p>
                            <p className="text-sm font-bold text-gray-900">{lastExtracted.gstin || "—"}</p>
                        </div>
                        <div>
                            <p className="text-xs text-gray-400 font-semibold mb-1 uppercase">Cess Amount</p>
                            <p className="text-sm font-bold text-gray-900">
                                ₹ {lastExtracted.cess_amount?.toLocaleString(undefined, { minimumFractionDigits: 2 })}
                            </p>
                        </div>
                    </div>
                </div>
            )}
        </div>
    );
}

/* ============================================================
   Chat Page (with embedded toggle)
   ============================================================ */
function ChatPage() {
    const [messages, setMessages] = useState([])
    const [input, setInput] = useState('')
    const [loading, setLoading] = useState(false)
    const [conversationHistory, setConversationHistory] = useState([])
    const [chatHistory, setChatHistory] = useState([])
    const [currentChatId, setCurrentChatId] = useState(null)
    const [activeView, setActiveView] = useState('chat') // 'chat' | 'bills' | 'gst'
    const messagesEndRef = useRef(null)
    const navigate = useNavigate();

    const scrollToBottom = () => {
        messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' })
    }

    useEffect(() => { scrollToBottom() }, [messages])

    useEffect(() => {
        const savedHistory = localStorage.getItem('chatHistory')
        if (savedHistory) {
            try { setChatHistory(JSON.parse(savedHistory)) }
            catch (e) { console.error('Error loading chat history:', e) }
        }
    }, [])

    useEffect(() => {
        if (chatHistory.length > 0) localStorage.setItem('chatHistory', JSON.stringify(chatHistory))
    }, [chatHistory])

    const sendMessage = async (e) => {
        e.preventDefault()
        if (!input.trim() || loading) return
        const userMessage = input.trim()
        setInput('')
        setLoading(true)

        const newUserMessage = { role: 'user', content: userMessage, timestamp: new Date() }
        setMessages(prev => [...prev, newUserMessage])
        const updatedHistory = [...conversationHistory, { role: 'user', content: userMessage }]

        try {
            const response = await axios.post(`${API_BASE_URL}/chat`, {
                message: userMessage, conversation_history: updatedHistory
            })
            const botMessage = {
                role: 'assistant', content: response.data.response,
                relevantSections: response.data.relevant_sections,
                relevantArticles: response.data.relevant_articles,
                confidence: response.data.confidence, timestamp: new Date()
            }
            const updatedMessages = [...messages, newUserMessage, botMessage]
            setMessages(updatedMessages)
            const finalHistory = [...updatedHistory, { role: 'assistant', content: response.data.response }]
            setConversationHistory(finalHistory)

            if (!currentChatId) {
                const newChatId = Date.now().toString()
                setCurrentChatId(newChatId)
                setChatHistory(prev => [{ id: newChatId, title: userMessage.substring(0, 50) + (userMessage.length > 50 ? '...' : ''), messages: updatedMessages, timestamp: new Date() }, ...prev])
            } else {
                setChatHistory(prev => prev.map(chat => chat.id === currentChatId ? { ...chat, messages: updatedMessages } : chat))
            }
        } catch (error) {
            console.error('Error:', error)
            setMessages(prev => [...prev, { role: 'assistant', content: 'Sorry, I encountered an error. Please try again.', error: true, timestamp: new Date() }])
        } finally { setLoading(false) }
    }

    const clearChat = () => { setMessages([]); setConversationHistory([]); setCurrentChatId(null) }

    const loadChat = (chatId) => {
        const chat = chatHistory.find(c => c.id === chatId)
        if (chat) {
            setMessages(chat.messages); setCurrentChatId(chatId)
            setConversationHistory(chat.messages.map(msg => ({ role: msg.role, content: msg.content })))
            setActiveView('chat')
        }
    }

    const deleteChat = (chatId, e) => {
        e.stopPropagation()
        setChatHistory(prev => prev.filter(chat => chat.id !== chatId))
        if (currentChatId === chatId) clearChat()
    }

    const startNewChat = () => { clearChat(); setActiveView('chat') }

    const handleKeyPress = (e) => {
        if (e.key === 'Enter' && !e.shiftKey) { e.preventDefault(); sendMessage(e) }
    }

    const promptSuggestions = [
        { title: "Tax Deductions", description: "What deductions can I claim under Section 80C?", icon: PiggyBank, color: "bg-blue-50 text-blue-600" },
        { title: "Tax Calculation", description: "How is income tax calculated for salaried employees?", icon: Calculator, color: "bg-purple-50 text-purple-600" },
        { title: "Capital Gains", description: "What are the rules for long-term capital gains tax?", icon: TrendingUp, color: "bg-green-50 text-green-600" },
    ]

    const handlePromptClick = (prompt) => { setInput(prompt.description) }

    const footerItems = [
        { title: "Settings", icon: Settings },
        { title: "Help Center", icon: HelpCircle },
        { title: "Sign Out", icon: LogOut },
    ]

    return (
        <div className="flex h-screen w-full bg-white overflow-hidden">
            {/* Sidebar */}
            <aside className="w-[280px] flex-shrink-0 border-r border-gray-100 flex flex-col bg-[#FAFAFA] h-full pb-4 pt-4">

                {/* Top Section */}
                <div className="px-4">
                    {/* New chat button */}
                    <button
                        onClick={startNewChat}
                        className="w-full flex items-center justify-center gap-2 bg-[#171717] hover:bg-black text-white rounded-xl py-3 px-4 font-semibold text-sm transition-colors mb-4 shadow-sm"
                    >
                        <Plus className="w-5 h-5" />
                        New chat
                    </button>


                    {/* Navigation Items */}
                    <nav className="flex flex-col gap-1 mb-8">
                        <button
                            onClick={() => setActiveView('chat')}
                            className={`flex items-center gap-3.5 w-full px-3 py-2.5 text-sm font-bold rounded-xl transition-all ${activeView === 'chat' ? 'bg-white shadow-[0_2px_8px_rgba(0,0,0,0.04)] text-gray-900 border border-gray-100/60' : 'text-gray-700 hover:bg-gray-100/50'}`}
                        >
                            <Globe className="w-5 h-5 text-gray-500 stroke-[2]" />
                            Chat
                        </button>
                        <button
                            onClick={() => setActiveView('bills')}
                            className={`flex items-center gap-3.5 w-full px-3 py-2.5 text-sm font-bold rounded-xl transition-all ${activeView === 'bills' ? 'bg-white shadow-[0_2px_8px_rgba(0,0,0,0.04)] text-gray-900 border border-gray-100/60' : 'text-gray-700 hover:bg-gray-100/50'}`}
                        >
                            <Library className="w-5 h-5 text-gray-500 stroke-[2]" />
                            Bills
                        </button>
                        <button
                            onClick={() => setActiveView('gst')}
                            className={`flex items-center gap-3.5 w-full px-3 py-2.5 text-sm font-bold rounded-xl transition-all ${activeView === 'gst' ? 'bg-white shadow-[0_2px_8px_rgba(0,0,0,0.04)] text-gray-900 border border-gray-100/60' : 'text-gray-700 hover:bg-gray-100/50'}`}
                        >
                            <Folder className="w-5 h-5 text-gray-500 stroke-[2]" />
                            GST Invoice
                        </button>

                    </nav>
                </div>

                <div className="px-4 mb-4">
                    <div className="w-full h-px bg-gray-200"></div>
                </div>

                {/* Chat History Section */}
                <div className="flex-1 overflow-y-auto px-2 pb-4 scrollbar-hide">
                    {chatHistory.length === 0 ? (
                        <div className="px-4 py-8 text-sm text-gray-400 text-center font-medium">
                            No previous history
                        </div>
                    ) : (
                        <div>
                            <div className="px-3 mb-2 text-xs font-semibold text-gray-400">
                                Today
                            </div>
                            <div className="flex flex-col gap-0.5">
                                {chatHistory.map((chat) => (
                                    <div
                                        key={chat.id}
                                        onClick={() => loadChat(chat.id)}
                                        className={`flex items-center group justify-between w-full px-3 py-2 text-[13px] font-medium rounded-lg cursor-pointer transition-colors ${currentChatId === chat.id ? 'bg-gray-200/50 text-gray-900' : 'text-gray-600 hover:bg-gray-200/50'}`}
                                    >
                                        <span className="truncate pr-2 w-[180px]">{chat.title}</span>
                                        <button
                                            className="opacity-0 group-hover:opacity-100 text-gray-400 hover:text-red-500 transition-opacity"
                                            onClick={(e) => deleteChat(chat.id, e)}
                                        >
                                            <Trash2 className="w-3.5 h-3.5" />
                                        </button>
                                    </div>
                                ))}
                            </div>
                        </div>
                    )}
                </div>

                {/* User Profile Footer */}
                <div className="px-4 mt-auto pt-4 relative">
                    <div className="absolute top-0 left-0 w-full h-12 bg-gradient-to-t from-[#FAFAFA] to-transparent pointer-events-none -translate-y-full"></div>
                    <div className="flex items-center justify-between w-full p-2 bg-white border border-gray-200 rounded-xl shadow-sm cursor-pointer hover:border-gray-300 transition-colors">
                        <div className="flex items-center gap-3 w-full overflow-hidden">
                            <div className="flex flex-col min-w-0 pr-2">
                                <span className="text-sm font-bold text-gray-900 truncate">Venkatesh</span>
                                <span className="text-[11px] font-medium text-gray-400 truncate">Venkatesh@gmail.com</span>
                            </div>
                        </div>
                        <LogOut className="w-4 h-4 text-gray-400 flex-shrink-0 mr-1" />
                    </div>
                </div>
            </aside>

            {/* Main Content */}
            <main className="flex-1 flex flex-col">
                {/* Header */}
                <header className="header">
                    <div className="flex items-center gap-3">
                        <h1 className="header-title">
                            {activeView === 'bills' ? 'Bill Scanner' : activeView === 'gst' ? 'GST Invoices' : currentChatId ? 'Conversation' : 'Legal Lens'}
                        </h1>
                        {activeView === 'bills' && (
                            <span className="px-2.5 py-0.5 rounded-full bg-amber-50 text-amber-600 text-xs font-bold border border-amber-200/60">
                                OCR
                            </span>
                        )}
                        {activeView === 'gst' && (
                            <span className="px-2.5 py-0.5 rounded-full bg-indigo-50 text-indigo-600 text-xs font-bold border border-indigo-200/60">
                                EXCEL
                            </span>
                        )}
                        {activeView === 'chat' && currentChatId && (
                            <span className="px-2.5 py-0.5 rounded-full bg-blue-50 text-blue-600 text-xs font-bold">
                                Active
                            </span>
                        )}
                    </div>
                </header>

                <div className="flex-1 overflow-y-auto">
                    {activeView === 'bills' ? (
                        /* ---- Bill Scanner View ---- */
                        <BillScannerPanel />
                    ) : activeView === 'gst' ? (
                        /* ---- GST Scanner View ---- */
                        <GSTInvoicePanel />
                    ) : (
                        /* ---- Chat View ---- */
                        <div className="main-content-wrapper">
                            {messages.length === 0 ? (
                                <div className="flex flex-col justify-center min-h-[calc(100vh-160px)]">
                                    <div className="welcome-section">
                                        <h2 className="welcome-title">Welcome to Legal Lens!</h2>
                                    </div>

                                    <div className="chat-input-card">
                                        <div className="chat-input-wrapper">
                                            <input type="text" value={input} onChange={(e) => setInput(e.target.value)}
                                                onKeyPress={handleKeyPress} placeholder="Ask about any tax law, section, or deduction..."
                                                className="chat-input" disabled={loading} />
                                        </div>
                                        <div className="chat-input-actions flex justify-end">
                                            <button onClick={sendMessage} className="send-button" disabled={loading || !input.trim()}>
                                                <Send className="h-4 w-4 text-white" />
                                            </button>
                                        </div>
                                    </div>

                                    <div className="feature-cards-grid">
                                        {promptSuggestions.map((prompt, idx) => (
                                            <div key={idx} className="feature-card" onClick={() => handlePromptClick(prompt)}>
                                                <div className={`feature-icon-wrapper ${prompt.color}`}>
                                                    <prompt.icon className="h-5 w-5" />
                                                </div>
                                                <div className="feature-title">{prompt.title}</div>
                                                <div className="feature-description">{prompt.description}</div>
                                            </div>
                                        ))}
                                    </div>
                                </div>
                            ) : (
                                <>
                                    <div className="messages-area">
                                        <div className="messages-header">
                                            <h3>Conversation</h3>
                                            <button onClick={clearChat} className="clear-button">Clear Chat</button>
                                        </div>
                                        <div className="messages-list">
                                            {messages.map((msg, idx) => (
                                                <div key={idx} className={`message ${msg.role === 'user' ? 'user-message' : 'bot-message'}`}>
                                                    <div className="message-content">
                                                        <div className="message-text">{msg.content}</div>
                                                        {msg.relevantSections?.length > 0 && (
                                                            <div className="relevant-sections">
                                                                <h4 className="relevant-sections-title">📚 Relevant Sections:</h4>
                                                                {msg.relevantSections.map((section, secIdx) => (
                                                                    <div key={secIdx} className="section-card">
                                                                        <div className="section-header">
                                                                            <strong>Section {section.section}</strong>
                                                                            <span className="similarity-score">{(section.similarity_score * 100).toFixed(1)}%</span>
                                                                        </div>
                                                                        <div className="section-title">{section.title}</div>
                                                                        <div className="section-summary">{section.summary}</div>
                                                                    </div>
                                                                ))}
                                                            </div>
                                                        )}
                                                        {msg.relevantArticles?.length > 0 && (
                                                            <div className="relevant-sections" style={{ marginTop: '16px' }}>
                                                                <h4 className="relevant-sections-title" style={{ color: '#6366f1' }}>📰 Related Articles:</h4>
                                                                {msg.relevantArticles.map((art, artIdx) => (
                                                                    <div key={artIdx} className="section-card" style={{ borderLeftColor: '#6366f1' }}>
                                                                        <div className="section-header"><strong style={{ fontSize: '0.9rem' }}>{art.title}</strong></div>
                                                                        <div style={{ fontSize: '0.75rem', color: '#9ca3af', marginBottom: '4px' }}>{art.date} • {art.author}</div>
                                                                        <div className="section-summary" style={{ marginBottom: '6px' }}>{art.snippet}</div>
                                                                        <div style={{ fontSize: '0.75rem', color: '#6366f1', fontWeight: 600 }}>Refers to Section {art.related_section}</div>
                                                                    </div>
                                                                ))}
                                                            </div>
                                                        )}
                                                    </div>
                                                </div>
                                            ))}
                                            {loading && (
                                                <div className="message bot-message">
                                                    <div className="message-content">
                                                        <div className="loading-dots"><span></span><span></span><span></span></div>
                                                    </div>
                                                </div>
                                            )}
                                            <div ref={messagesEndRef} />
                                        </div>
                                    </div>

                                    <div className="chat-input-card" style={{ marginTop: '20px' }}>
                                        <div className="chat-input-wrapper">
                                            <input type="text" value={input} onChange={(e) => setInput(e.target.value)}
                                                onKeyPress={handleKeyPress} placeholder="Ask about any tax law, section, or deduction..."
                                                className="chat-input" disabled={loading} />
                                        </div>
                                        <div className="chat-input-actions">
                                            <div className="chat-input-icons">
                                                <button className="icon-button"><Paperclip className="h-4 w-4" /></button>
                                                <button className="icon-button"><Image className="h-4 w-4" /></button>
                                                <button className="icon-button"><File className="h-4 w-4" /></button>
                                            </div>
                                            <button onClick={sendMessage} className="send-button" disabled={loading || !input.trim()}>
                                                <Send className="h-4 w-4 text-white" />
                                            </button>
                                        </div>
                                    </div>
                                </>
                            )}
                        </div>
                    )}
                </div>
            </main>
        </div>
    )
}

export default ChatPage
