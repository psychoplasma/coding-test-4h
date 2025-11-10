"use client";

import { useEffect, useState } from "react";
import Link from "next/link";
import { useRouter } from "next/navigation";

interface Document {
  id: number;
  filename: string;
  upload_date: string;
  status: string;
  total_pages: number;
  text_chunks: number;
  error_message?: string;
  images: any[];
  tables: any[];
}

const BACKEND_URL = process.env.BACKEND_URL || "http://localhost:8000";
const API_URL = `${BACKEND_URL}/api`;

export default function Home() {
  const router = useRouter();
  const [documents, setDocuments] = useState<Document[]>([]);
  const [loading, setLoading] = useState(true);
  const [selectedIds, setSelectedIds] = useState<Set<number>>(new Set());

  useEffect(() => {
    fetchDocuments();
  }, []);

  const fetchDocuments = async () => {
    try {
      const response = await fetch(`${API_URL}/documents?limit=100`);
      const data = await response.json();
      setDocuments(data.items || []);
    } catch (error) {
      console.error("Error fetching documents:", error);
    } finally {
      setLoading(false);
    }
  };

  const toggleSelection = (id: number) => {
    const newSelected = new Set(selectedIds);
    if (newSelected.has(id)) {
      newSelected.delete(id);
    } else {
      newSelected.add(id);
    }
    setSelectedIds(newSelected);
  };

  const toggleSelectAll = () => {
    if (selectedIds.size === documents.length) {
      setSelectedIds(new Set());
    } else {
      setSelectedIds(new Set(documents.map((doc) => doc.id)));
    }
  };

  const deleteSelectedDocuments = async () => {
    if (!confirm(`Are you sure you want to delete ${selectedIds.size} document(s)?`)) return;

    try {
      await Promise.all(
        Array.from(selectedIds).map((id) =>
          fetch(`${API_URL}/documents/${id}`, { method: "DELETE" })
        )
      );
      setSelectedIds(new Set());
      fetchDocuments();
    } catch (error) {
      console.error("Error deleting documents:", error);
    }
  };

  const handleChat = () => {
    const ids = Array.from(selectedIds).join(",");
    router.push(`/chat?documents=${ids}`);
  };

  const handleCardClick = (id: number) => {
    router.push(`/documents/${id}`);
  };

  return (
    <div className="px-4 sm:px-0">
      <div className="flex justify-between items-center mb-6">
        <h1 className="text-3xl font-bold text-gray-900">My Documents</h1>
        <div className="flex gap-2">
          {selectedIds.size > 0 && (
            <>
              <button
                onClick={handleChat}
                className="bg-green-600 text-white px-4 py-2 rounded-lg hover:bg-green-700"
              >
                Chat ({selectedIds.size})
              </button>
              <button
                onClick={deleteSelectedDocuments}
                className="bg-red-600 text-white px-4 py-2 rounded-lg hover:bg-red-700"
              >
                Delete ({selectedIds.size})
              </button>
            </>
          )}
          <Link
            href="/upload"
            className="bg-blue-600 text-white px-4 py-2 rounded-lg hover:bg-blue-700"
          >
            Upload New Document
          </Link>
        </div>
      </div>

      {loading ? (
        <div className="text-center py-12">
          <div className="inline-block h-8 w-8 animate-spin rounded-full border-4 border-solid border-blue-600 border-r-transparent"></div>
          <p className="mt-2 text-gray-600">Loading documents...</p>
        </div>
      ) : documents.length === 0 ? (
        <div className="text-center py-12 bg-white rounded-lg shadow">
          <p className="text-gray-500">No documents uploaded yet.</p>
          <Link
            href="/upload"
            className="mt-4 inline-block text-blue-600 hover:text-blue-700"
          >
            Upload your first document →
          </Link>
        </div>
      ) : (
        <div className="bg-white shadow overflow-hidden sm:rounded-md">
          <ul className="divide-y divide-gray-200">
            <li className="px-4 py-3 sm:px-6 bg-gray-50 border-b border-gray-200">
              <div className="flex items-center">
                <input
                  type="checkbox"
                  checked={selectedIds.size === documents.length && documents.length > 0}
                  onChange={toggleSelectAll}
                  className="h-4 w-4 text-blue-600 rounded cursor-pointer"
                />
                <span className="ml-3 text-sm text-gray-600">
                  {selectedIds.size > 0
                    ? `${selectedIds.size} selected`
                    : "Select all"}
                </span>
              </div>
            </li>
            {documents.map((doc) => (
              <li key={doc.id}>
                <div
                  className="px-4 py-4 flex items-center sm:px-6 hover:bg-blue-50 cursor-pointer transition-colors"
                  onClick={() => handleCardClick(doc.id)}
                >
                  <input
                    type="checkbox"
                    checked={selectedIds.has(doc.id)}
                    onChange={(e) => {
                      e.stopPropagation();
                      toggleSelection(doc.id);
                    }}
                    className="h-4 w-4 text-blue-600 rounded cursor-pointer"
                    onClick={(e) => e.stopPropagation()}
                  />
                  <div className="min-w-0 flex-1 ml-4 sm:flex sm:items-center sm:justify-between">
                    <div className="truncate">
                      <div className="flex text-sm">
                        <p className="font-medium text-blue-600 truncate">
                          {doc.filename}
                        </p>
                        <p className="ml-2 flex-shrink-0 font-normal text-gray-500">
                          <span className={`px-2 inline-flex text-xs leading-5 font-semibold rounded-full ${
                            doc.status === 'completed' ? 'bg-green-100 text-green-800' :
                            doc.status === 'processing' ? 'bg-yellow-100 text-yellow-800' :
                            doc.status === 'error' ? 'bg-red-100 text-red-800' :
                            'bg-gray-100 text-gray-800'
                          }`}>
                            {doc.status}
                          </span>
                        </p>
                      </div>
                      <div className="mt-2 flex">
                        <div className="flex items-center text-sm text-gray-500">
                          <p>
                            {doc.total_pages} pages • {doc.text_chunks} chunks • {doc.images.length} images • {doc.tables.length} tables
                          </p>
                        </div>
                      </div>
                    </div>
                  </div>
                </div>
              </li>
            ))}
          </ul>
        </div>
      )}
    </div>
  );
}
