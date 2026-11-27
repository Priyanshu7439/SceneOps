
import { useCallback, useEffect, useState } from "react";
import type { ReactNode } from "react";
import {
  Activity,
  Ambulance,
  Clock3,
  LayoutDashboard,
  Users,
  Radio,
  Settings,
  Siren,
  Pause,
  Play,
  RotateCcw,
  Search,
  Bell,
  ChevronRight,
  HeartPulse,
  MapPin,
  FastForward,
} from "lucide-react";
import "./App.css";

type SimulationStatus = "created" | "running" | "paused" | "completed";

interface CreateSessionResponse {
  session_id: string;
  scenario_id: string;
  title: string;
  description: string;
  patient_count: number;
  status: SimulationStatus;
}

interface SessionSnapshot {
  session_id: string;
  status: SimulationStatus;
  clock: {
    elapsed_seconds: number;
    tick_count: number;
  };
  patient_count: number;
  assessed_patient_count: number;
  allocated_patient_count: number;
  available_ambulance_count: number;
  event_count: number;
}

interface Patient {
  patient_id: string;
  assessment_status: string;
  triage_category: string;
  ambulance_id: string | null;
  transport_status: string;
}

interface PatientListResponse {
  patients: Patient[];
}

type WorkspacePage = "overview" | "patients" | "resources" | "timeline";

interface Ambulance {
  ambulance_id: string;
  status: string;
  assigned_patient_id: string | null;
}

interface EventRecord {
  event_type: string;
  simulation_second: number;
  description: string;
  patient_id: string | null;
}

type ObservationType =
  | "respiratory_rate"
  | "radial_pulse"
  | "capillary_refill"
  | "mental_status"
  | "ambulatory_status"
  | "visible_hemorrhage"
  | "reported_pain"
  | "reported_symptom"
  | "follows_commands";

interface PatientObservation {
  patient_id: string;
  observation_type: ObservationType;
  value: string;
  observed_at_seconds: number;
}

interface NarrationResponse {
  agent_name: string;
  content: string;
}

const OBSERVATION_OPTIONS: { value: ObservationType; label: string }[] = [
  { value: "respiratory_rate", label: "Respiratory rate" },
  { value: "radial_pulse", label: "Radial pulse" },
  { value: "capillary_refill", label: "Capillary refill" },
  { value: "mental_status", label: "Mental status" },
  { value: "ambulatory_status", label: "Ambulatory status" },
  { value: "visible_hemorrhage", label: "Visible hemorrhage" },
  { value: "follows_commands", label: "Follows commands" },
];

async function apiRequest<T>(
  path: string,
  options: RequestInit = {},
): Promise<T> {
  const response = await fetch(path, {
    ...options,
    headers: {
      "Content-Type": "application/json",
      ...options.headers,
    },
  });

  if (!response.ok) {
    const body = await response.json().catch(() => null);
    throw new Error(
      body?.detail
        ? String(body.detail)
        : `Request failed with status ${response.status}`,
    );
  }

  if (response.status === 204) {
    return undefined as T;
  }

  return response.json() as Promise<T>;
}

function formatClock(totalSeconds: number): string {
  const hours = Math.floor(totalSeconds / 3600);
  const minutes = Math.floor((totalSeconds % 3600) / 60);
  const seconds = totalSeconds % 60;

  return [hours, minutes, seconds]
    .map((value) => String(value).padStart(2, "0"))
    .join(":");
}

function formatLabel(value: string): string {
  return value
    .replaceAll("_", " ")
    .replace(/\b\w/g, (letter) => letter.toUpperCase());
}

function App() {
  const [sessionId, setSessionId] = useState<string | null>(null);
  const [scenarioId, setScenarioId] = useState("MCI-001");
  const [status, setStatus] = useState<SimulationStatus | "not_started">(
    "not_started",
  );
  const [snapshot, setSnapshot] = useState<SessionSnapshot | null>(null);
  const [patients, setPatients] = useState<Patient[]>([]);
  const [activePage, setActivePage] = useState<WorkspacePage>("overview");
  const [ambulances, setAmbulances] = useState<Ambulance[]>([]);
  const [events, setEvents] = useState<EventRecord[]>([]);
  const [observations, setObservations] = useState<PatientObservation[]>([]);
  const [sceneNarration, setSceneNarration] = useState("");
  const [temporalNarration, setTemporalNarration] = useState("");
  const [narrationBusy, setNarrationBusy] = useState(false);
  const [narrationError, setNarrationError] = useState("");
  const [selectedPatientId, setSelectedPatientId] = useState<string | null>(null);
  const [searchQuery, setSearchQuery] = useState("");
  const [backendOnline, setBackendOnline] = useState(false);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState("");

  const loadDashboard = useCallback(async (id: string) => {
    // Load each dashboard resource independently so one failed endpoint does
    // not prevent the other successful sections from refreshing.
    const [sessionResult, patientResult, resourceResult, eventResult] =
      await Promise.allSettled([
        apiRequest<SessionSnapshot>(`/sessions/${id}`),
        apiRequest<PatientListResponse>(`/sessions/${id}/patients`),
        apiRequest<{ ambulances: Ambulance[] }>(`/sessions/${id}/resources`),
        apiRequest<{ events: EventRecord[] }>(`/sessions/${id}/events`),
      ] as const);

    const failures: string[] = [];
    const describeFailure = (endpoint: string, reason: unknown) => {
      const message = reason instanceof Error ? reason.message : String(reason);
      failures.push(`${endpoint}: ${message}`);
    };

    if (sessionResult.status === "fulfilled") {
      setSnapshot(sessionResult.value);
      setStatus(sessionResult.value.status);
    } else {
      describeFailure(`/sessions/${id}`, sessionResult.reason);
    }

    if (patientResult.status === "fulfilled") {
      setPatients(patientResult.value.patients);
    } else {
      describeFailure(`/sessions/${id}/patients`, patientResult.reason);
    }

    if (resourceResult.status === "fulfilled") {
      setAmbulances(resourceResult.value.ambulances);
    } else {
      describeFailure(`/sessions/${id}/resources`, resourceResult.reason);
    }

    if (eventResult.status === "fulfilled") {
      setEvents(eventResult.value.events);
    } else {
      describeFailure(`/sessions/${id}/events`, eventResult.reason);
    }

    if (failures.length > 0) {
      const message = `Dashboard refresh incomplete — ${failures.join("; ")}`;
      setError(message);
      throw new Error(message);
    }

    // Clear stale 502/error banners after a completely successful refresh.
    setError("");
  }, []);

  // Verify backend connectivity on startup.
  useEffect(() => {
    let active = true;

    apiRequest<{ status: string }>("/health")
      .then(() => {
        if (active) setBackendOnline(true);
      })
      .catch(() => {
        if (active) setBackendOnline(false);
      });

    return () => {
      active = false;
    };
  }, []);

  // Refresh session data periodically.
  useEffect(() => {
    if (!sessionId) return;

    const interval = window.setInterval(() => {
      loadDashboard(sessionId).catch((err: Error) => {
        setError(err.message);
      });
    }, 3000);

    return () => window.clearInterval(interval);
  }, [sessionId, loadDashboard]);

  const handleStart = async () => {
    setBusy(true);
    setError("");

    try {
      let activeSessionId = sessionId;

      if (!activeSessionId) {
        const created = await apiRequest<CreateSessionResponse>(
          "/sessions",
          {
            method: "POST",
            body: JSON.stringify({ scenario_index: 1 }),
          },
        );

        activeSessionId = created.session_id;
        setSessionId(activeSessionId);
        setScenarioId(created.scenario_id);
      }

      await apiRequest(
        `/sessions/${activeSessionId}/start`,
        { method: "POST" },
      );

      await loadDashboard(activeSessionId);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Unable to start simulation.");
    } finally {
      setBusy(false);
    }
  };

  const handlePause = async () => {
    if (!sessionId) return;

    setBusy(true);
    setError("");

    try {
      await apiRequest(`/sessions/${sessionId}/pause`, {
        method: "POST",
      });

      await loadDashboard(sessionId);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Unable to pause simulation.");
    } finally {
      setBusy(false);
    }
  };

  const handleResume = async () => {
    if (!sessionId) return;

    setBusy(true);
    setError("");

    try {
      await apiRequest(`/sessions/${sessionId}/resume`, {
        method: "POST",
      });

      await loadDashboard(sessionId);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Unable to resume simulation.");
    } finally {
      setBusy(false);
    }
  };

  const handleAdvance = async () => {
    if (!sessionId) return;

    setBusy(true);
    setError("");

    try {
      await apiRequest(`/sessions/${sessionId}/advance`, {
        method: "POST",
        body: JSON.stringify({ seconds: 60 }),
      });

      await loadDashboard(sessionId);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Unable to advance simulation.");
    } finally {
      setBusy(false);
    }
  };
  const handleAssessPatient = async () => {
    if (!sessionId || !selectedPatientId) return;

    setBusy(true);
    setError("");

    try {
      await apiRequest(
        `/sessions/${sessionId}/patients/${selectedPatientId}/assessment`,
        { method: "POST" },
      );

      await loadDashboard(sessionId);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Assessment failed.");
    } finally {
      setBusy(false);
    }
  };

  const handleRecordObservation = async (observationType: ObservationType) => {
    if (!sessionId || !selectedPatientId) return;

    setBusy(true);
    setError("");
    setNarrationError("");

    try {
      const observation = await apiRequest<PatientObservation>(
        `/sessions/${sessionId}/patients/${selectedPatientId}/observations`,
        {
          method: "POST",
          body: JSON.stringify({ observation_type: observationType }),
        },
      );
      setObservations((current) => [...current, observation]);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Unable to record observation.");
    } finally {
      setBusy(false);
    }
  };

  const handleGenerateNarration = async (temporal = false) => {
    if (!sessionId) return;

    setNarrationBusy(true);
    setNarrationError("");

    try {
      const path = temporal
        ? `/sessions/${sessionId}/temporal-narration`
        : `/sessions/${sessionId}/narration`;
      const response = await apiRequest<NarrationResponse>(path, {
        method: "POST",
        body: JSON.stringify({
          message: temporal
            ? "Summarize the chronological changes and unchanged findings using only the recorded learner observations. Do not infer unobserved events or clinical decisions."
            : "Summarize the currently recorded learner observations by patient. Use only the supplied observations and do not infer unobserved events or clinical decisions.",
        }),
      });

      if (temporal) {
        setTemporalNarration(response.content);
      } else {
        setSceneNarration(response.content);
      }
    } catch (err) {
      setNarrationError(
        err instanceof Error ? err.message : "Unable to generate narration.",
      );
    } finally {
      setNarrationBusy(false);
    }
  };

  const handleAllocateAmbulance = async () => {
    if (!sessionId || !selectedPatientId) return;

    setBusy(true);
    setError("");

    try {
      await apiRequest(
        `/sessions/${sessionId}/patients/${selectedPatientId}/allocation`,
        { method: "POST" },
      );

      await loadDashboard(sessionId);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Allocation failed.");
    } finally {
      setBusy(false);
    }
  };

  const handleStartTransport = async () => {
    if (!sessionId || !selectedPatientId) return;

    setBusy(true);
    setError("");

    try {
      await apiRequest(
        `/sessions/${sessionId}/patients/${selectedPatientId}/transport`,
        {
          method: "POST",
          body: JSON.stringify({ travel_duration_seconds: 300 }),
        },
      );

      await loadDashboard(sessionId);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Transport failed.");
    } finally {
      setBusy(false);
    }
  };

const handleReset = async () => {
  if (!sessionId) return;

  const confirmed = window.confirm(
    "Complete and delete the current simulation session?"
  );

  if (!confirmed) return;

  setBusy(true);
  setError("");

  try {
    // A running or paused session must be completed first.
    if (!isCompleted) {
      await apiRequest(`/sessions/${sessionId}/complete`, {
        method: "POST",
      });
    }

    // Delete the session after successful completion.
    await apiRequest(`/sessions/${sessionId}`, {
      method: "DELETE",
    });

    // Restore the initial dashboard state.
    setSessionId(null);
    setScenarioId("MCI-001");
    setSnapshot(null);
    setPatients([]);
    setAmbulances([]);
    setEvents([]);
    setObservations([]);
    setSceneNarration("");
    setTemporalNarration("");
    setNarrationError("");
    setSelectedPatientId(null);
    setSearchQuery("");
    setStatus("not_started");
  } catch (err) {
    setError(
      err instanceof Error
        ? err.message
        : "Unable to reset simulation."
    );
  } finally {
    setBusy(false);
  }
};


  const elapsedSeconds = snapshot?.clock?.elapsed_seconds ?? 0;

  const filteredPatients = patients.filter((patient) =>
    patient.patient_id.toLowerCase().includes(searchQuery.toLowerCase()),
  );

  const selectedPatient =
    patients.find((patient) => patient.patient_id === selectedPatientId) ?? null;

  const isRunning = status === "running";
  const isPaused = status === "paused";
  const isCompleted = status === "completed";

  const statusLabel =
    status === "not_started"
      ? "Simulation not started"
      : `Simulation ${formatLabel(status).toLowerCase()}`;

  return (
    <div className="app-shell">
      <aside className="sidebar">
        <div className="brand">
          <div className="brand-icon"><Siren size={22} /></div>
          <div>
            <h2>SCENEOPS</h2>
            <span>INCIDENT SIMULATOR</span>
          </div>
        </div>

        <div className="nav-label">WORKSPACE</div>
        <nav>
          <button
            className={`nav-item ${activePage === "overview" ? "active" : ""}`}
            onClick={() => setActivePage("overview")}
          >
            <LayoutDashboard size={18} /> Overview
          </button>
          <button
            className={`nav-item ${activePage === "patients" ? "active" : ""}`}
            onClick={() => setActivePage("patients")}
          >
            <Users size={18} /> Patient registry
          </button>
          <button
            className={`nav-item ${activePage === "resources" ? "active" : ""}`}
            onClick={() => setActivePage("resources")}
          >
            <Ambulance size={18} /> Resources
          </button>
          <button
            className={`nav-item ${activePage === "timeline" ? "active" : ""}`}
            onClick={() => setActivePage("timeline")}
          >
            <Activity size={18} /> Event timeline
          </button>
        </nav>

        <div className="sidebar-bottom">
          <div className="system-status">
            <span className="live-dot" />
            <div>
              <strong>System {backendOnline ? "operational" : "offline"}</strong>
              <small>
                {backendOnline ? "Backend connected" : "Backend unavailable"}
              </small>
            </div>
          </div>
          <a className="nav-item"><Settings size={18} /> Settings</a>
          <div className="version">SCENEOPS v0.1.0</div>
        </div>
      </aside>

      <main className="main-content">
        <header className="topbar">
          <div className="breadcrumb">
            Workspace <ChevronRight size={14} /> <strong>{formatLabel(activePage)}</strong>
          </div>
          <div className="top-actions">
            <span className="environment">
              <span className="live-dot" />
              {backendOnline ? "BACKEND CONNECTED" : "BACKEND OFFLINE"}
            </span>
            <button className="icon-button" aria-label="Notifications">
              <Bell size={18} />
            </button>
            <div className="avatar">SO</div>
          </div>
        </header>

        <section className="page-content">
          {activePage === "overview" && (
            <>
          <div className="page-heading">
            <div>
              <div className="eyebrow">INCIDENT COMMAND / DASHBOARD</div>
              <h1>Operations overview</h1>
              <p>Monitor incident activity, patient status and resource allocation.</p>
            </div>
            <div className="scenario-chip">
              <Radio size={15} /> {scenarioId}
              <span>{sessionId ? "LIVE SESSION" : "READY"}</span>
            </div>
          </div>

          <section className="simulation-bar">
            <div className="simulation-info">
              <span className="status-indicator" />
              <div>
                <strong>{statusLabel}</strong>
                <small>
                  {isRunning
                    ? "Simulation is active"
                    : isPaused
                      ? "Simulation is paused"
                      : isCompleted
                        ? "Simulation completed"
                        : "Awaiting operator command"}
                </small>
              </div>
            </div>

            <div className="simulation-clock">
              <Clock3 size={17} />
              <span>{formatClock(elapsedSeconds)}</span>
            </div>

            <div className="simulation-controls">
              <button
                className="control-button primary"
                onClick={handleStart}
                disabled={busy || !backendOnline || isRunning || isPaused || isCompleted}
              >
                <Play size={15} /> {busy ? "Working..." : "Start"}
              </button>

              {isPaused ? (
                <button
                  className="control-button primary"
                  onClick={handleResume}
                  disabled={busy}
                >
                  <Play size={15} /> Resume
                </button>
              ) : (
                <button
                  className="control-button"
                  onClick={handlePause}
                  disabled={busy || !isRunning}
                >
                  <Pause size={15} /> Pause
                </button>
              )}

              <button
                className="control-button"
                onClick={handleAdvance}
                disabled={busy || !isRunning}
                title="Advance simulation by 60 seconds"
              >
                <FastForward size={15} /> +60 sec
              </button>

              <button
                className="control-button"
                onClick={handleReset}
                disabled={busy || !sessionId}
              >
                <RotateCcw size={15} /> Reset
              </button>
            </div>
          </section>

          {error && (
            <div
              role="alert"
              style={{
                background: "#f8e4e0",
                color: "#a83d37",
                border: "1px solid #e8c4bc",
                borderRadius: "7px",
                padding: "12px 15px",
                marginBottom: "16px",
                fontSize: "12px",
              }}
            >
              <strong>API error:</strong> {error}
            </div>
          )}

          <section className="metrics-grid">
            <Metric
              icon={<Users />}
              label="Total patients"
              value={String(snapshot?.patient_count ?? 0)}
              note={snapshot ? "Live scenario population" : "Start a simulation"}
            />
            <Metric
              icon={<HeartPulse />}
              label="Assessed patients"
              value={String(snapshot?.assessed_patient_count ?? 0)}
              note="Simulator assessment count"
            />
            <Metric
              icon={<Ambulance />}
              label="Available ambulances"
              value={snapshot ? String(snapshot.available_ambulance_count) : "—"}
              note="Current resource availability"
            />
            <Metric
              icon={<Activity />}
              label="Recorded events"
              value={String(snapshot?.event_count ?? 0)}
              note="Simulator event count"
            />
          </section>

          <div className="dashboard-grid overview-dashboard-grid">
            <section className="panel patient-panel">
              <div className="panel-heading">
                <div>
                  <h3>Patient registry</h3>
                  <p>Live overview of incident patients</p>
                </div>
                <span className="record-count">{patients.length} PATIENTS</span>
              </div>

              <div className="table-toolbar">
                <div className="search-box">
                  <Search size={16} />
                  <input
                    value={searchQuery}
                    onChange={(event) => setSearchQuery(event.target.value)}
                    placeholder="Search patient ID..."
                    aria-label="Search patient ID"
                    style={{
                      border: "none",
                      outline: "none",
                      background: "transparent",
                      color: "inherit",
                      fontSize: "11px",
                      width: "150px",
                    }}
                  />
                </div>
                <span className="record-count">{filteredPatients.length} RESULTS</span>
              </div>

              <div className="patient-table-wrap overview-patient-table-wrap">
                <table className="patient-table">
                  <thead>
                    <tr>
                      <th>PATIENT ID</th>
                      <th>TRIAGE</th>
                      <th>ASSESSMENT</th>
                      <th>TRANSPORT</th>
                    </tr>
                  </thead>
                  <tbody>
                    {filteredPatients.map((patient) => (
                      <tr
                        key={patient.patient_id}
                        className={`patient-row ${
                          selectedPatientId === patient.patient_id ? "selected" : ""
                        }`}
                        onClick={() => setSelectedPatientId(patient.patient_id)}
                        onKeyDown={(event) => {
                          if (event.key === "Enter" || event.key === " ") {
                            event.preventDefault();
                            setSelectedPatientId(patient.patient_id);
                          }
                        }}
                        tabIndex={0}
                        aria-selected={selectedPatientId === patient.patient_id}
                        role="button"
                        aria-label={`View details for ${patient.patient_id}`}
                      >
                        <td className="patient-id">{patient.patient_id}</td>
                        <td>
                          <span className={`triage-badge ${patient.triage_category.toLowerCase()}`}>
                            {patient.triage_category === "pending_assessment"
                              ? "PENDING"
                              : patient.triage_category.toUpperCase()}
                          </span>
                        </td>
                        <td><span className="muted-status">{formatLabel(patient.assessment_status)}</span></td>
                        <td><span className="muted-status">{formatLabel(patient.transport_status)}</span></td>
                      </tr>
                    ))}
                    {filteredPatients.length === 0 && (
                      <tr>
                        <td colSpan={4} className="patient-table-empty">
                          {patients.length === 0
                            ? "Start a simulation to load patients."
                            : "No matching patients found."}
                        </td>
                      </tr>
                    )}
                  </tbody>
                </table>
              </div>

              <div className="table-footer">
                {sessionId
                  ? "Live data · Connected to SceneOps API"
                  : "No active session · Patient data will appear after starting"}
              </div>
            </section>

            <div className="overview-side-stack">
              <section className="panel patient-details-panel overview-details-panel">
                <div className="panel-heading">
                  <div>
                    <h3>Patient Details</h3>
                    <p>
                      {selectedPatient
                        ? "Selected patient information from the simulator"
                        : "Select a patient from the registry to view details"}
                    </p>
                  </div>
                  {selectedPatient && (
                    <button
                      type="button"
                      className="text-button"
                      onClick={() => setSelectedPatientId(null)}
                    >
                      Close
                    </button>
                  )}
                </div>

                {selectedPatient ? (
                  <>
                    <div className="patient-actions">
                      <button
                        type="button"
                        className="control-button"
                        onClick={handleAssessPatient}
                        disabled={
                          busy ||
                          !isRunning ||
                          selectedPatient.assessment_status === "assessed"
                        }
                      >
                        Assess Patient
                      </button>

                      <button
                        type="button"
                        className="control-button"
                        onClick={handleAllocateAmbulance}
                        disabled={
                          busy ||
                          !isRunning ||
                          selectedPatient.assessment_status !== "assessed" ||
                          !["red", "yellow", "green"].includes(selectedPatient.triage_category) ||
                          selectedPatient.ambulance_id !== null ||
                          selectedPatient.transport_status !== "not_allocated"
                        }
                      >
                        Allocate Ambulance
                      </button>

                      <button
                        type="button"
                        className="control-button primary"
                        onClick={handleStartTransport}
                        disabled={
                          busy ||
                          !isRunning ||
                          selectedPatient.transport_status !== "allocated"
                        }
                      >
                        Start Transport · 5 min
                      </button>
                    </div>

                    <div className="patient-details-grid overview-patient-details-grid">
                      <div className="detail-item">
                        <span>Patient ID</span>
                        <strong>{selectedPatient.patient_id}</strong>
                      </div>
                      <div className="detail-item">
                        <span>Triage category</span>
                        <strong>{formatLabel(selectedPatient.triage_category)}</strong>
                      </div>
                      <div className="detail-item">
                        <span>Assessment status</span>
                        <strong>{formatLabel(selectedPatient.assessment_status)}</strong>
                      </div>
                      <div className="detail-item">
                        <span>Ambulance</span>
                        <strong>{selectedPatient.ambulance_id ?? "Not allocated"}</strong>
                      </div>
                      <div className="detail-item">
                        <span>Transport status</span>
                        <strong>{formatLabel(selectedPatient.transport_status)}</strong>
                      </div>
                    </div>

                    <div className="observation-capture">
                      <div className="observation-capture-heading">
                        <strong>Record learner observations</strong>
                        <span>Simulator-derived findings · {observations.filter((item) => item.patient_id === selectedPatient.patient_id).length} recorded</span>
                      </div>
                      <div className="observation-actions">
                        {OBSERVATION_OPTIONS.map((option) => (
                          <button
                            key={option.value}
                            type="button"
                            className="control-button observation-button"
                            onClick={() => handleRecordObservation(option.value)}
                            disabled={busy || !isRunning}
                          >
                            {option.label}
                          </button>
                        ))}
                      </div>
                      <div className="observation-record-list">
                        {observations
                          .filter((item) => item.patient_id === selectedPatient.patient_id)
                          .slice(-5)
                          .reverse()
                          .map((item, index) => (
                            <div className="observation-record" key={`${item.observed_at_seconds}-${item.observation_type}-${index}`}>
                              <span>{formatLabel(item.observation_type)}</span>
                              <strong>{item.value}</strong>
                              <small>{formatClock(item.observed_at_seconds)}</small>
                            </div>
                          ))}
                        {observations.every((item) => item.patient_id !== selectedPatient.patient_id) && (
                          <p className="observation-empty">No observations recorded for this patient yet.</p>
                        )}
                      </div>
                    </div>
                  </>
                ) : (
                  <div className="patient-details-empty">
                    <Users size={23} />
                    <span>No patient selected</span>
                    <p>Choose any patient in the registry to assess, allocate and manage transport.</p>
                  </div>
                )}
              </section>

              <section className="panel intelligence-panel overview-intelligence-panel">
                <div className="panel-heading">
                  <div>
                    <h3>Incident intelligence</h3>
                    <p>Scene monitoring and narration</p>
                  </div>
                  <span className={backendOnline ? "online-tag" : "offline-tag"}>
                    {backendOnline ? "API CONNECTED" : "API OFFLINE"}
                  </span>
                </div>

                <div className="intelligence-actions">
                  <button
                    type="button"
                    className="control-button primary"
                    onClick={() => handleGenerateNarration(false)}
                    disabled={!sessionId || !isRunning || narrationBusy || observations.length === 0}
                  >
                    Generate scene narration
                  </button>
                  <button
                    type="button"
                    className="control-button"
                    onClick={() => handleGenerateNarration(true)}
                    disabled={!sessionId || !isRunning || narrationBusy || observations.length === 0}
                  >
                    Generate temporal narration
                  </button>
                  {narrationBusy && <span className="narration-loading">Generating with SceneOps model…</span>}
                </div>

                {narrationError && <p className="narration-error">{narrationError}</p>}

                <div className="narration-output">
                  <div>
                    <h4>Scene narration</h4>
                    <p>{sceneNarration || "Record learner observations, then generate scene narration."}</p>
                  </div>
                  <div>
                    <h4>Temporal narration</h4>
                    <p>{temporalNarration || "Record follow-up observations as simulation time advances, then generate a temporal summary."}</p>
                  </div>
                </div>

              </section>

              <section className="panel narration-integration-panel">
                <div className="panel-heading">
                  <div>
                    <h3>Narration integration</h3>
                    <p>Scene and temporal narration service</p>
                  </div>
                  <span className={backendOnline ? "online-tag" : "offline-tag"}>
                    {backendOnline ? "CONNECTED" : "OFFLINE"}
                  </span>
                </div>
                <div className="narration-integration-content">
                  <span className={`live-dot ${backendOnline ? "" : "muted-dot"}`} />
                  <div>
                    <strong>{observations.length} learner observations recorded</strong>
                    <p>
                      {sceneNarration || temporalNarration
                        ? "Narration generated from simulator-validated observations."
                        : "Narration is ready when observations have been recorded."}
                    </p>
                  </div>
                </div>
              </section>
            </div>
          </div>

          <section className="bottom-strip">
            <div>
              <MapPin size={16} />
              <span>Incident environment</span>
              <strong>{scenarioId}</strong>
            </div>
            <div>
              <Activity size={16} />
              <span>Simulation engine</span>
              <strong>{formatLabel(status)}</strong>
            </div>
            <div>
              <Radio size={16} />
              <span>Backend connection</span>
              <strong className={backendOnline ? "" : "disconnected"}>
                {backendOnline ? "Connected" : "Not connected"}
              </strong>
            </div>
          </section>
            </>
          )}

          {activePage === "patients" && (
            <>
              <div className="page-heading">
                <div>
                  <div className="eyebrow">INCIDENT COMMAND / PATIENTS</div>
                  <h1>Patient registry</h1>
                  <p>Review patients and their current simulator status.</p>
                </div>
                <span className="record-count">{patients.length} PATIENTS</span>
              </div>

              <section className="panel workspace-panel">
                <div className="table-toolbar">
                  <div className="search-box">
                    <Search size={16} />
                    <input
                      value={searchQuery}
                      onChange={(event) => setSearchQuery(event.target.value)}
                      placeholder="Search patient ID..."
                      aria-label="Search patient ID"
                    />
                  </div>
                  <span className="record-count">{filteredPatients.length} RESULTS</span>
                </div>

                <div className="patient-table-wrap">
                  <table className="patient-table">
                    <thead>
                      <tr>
                        <th>PATIENT ID</th>
                        <th>TRIAGE</th>
                        <th>ASSESSMENT</th>
                        <th>AMBULANCE</th>
                        <th>TRANSPORT</th>
                      </tr>
                    </thead>
                    <tbody>
                      {filteredPatients.map((patient) => (
                        <tr
                          key={patient.patient_id}
                          className={`patient-row ${
                            selectedPatientId === patient.patient_id ? "selected" : ""
                          }`}
                          onClick={() => setSelectedPatientId(patient.patient_id)}
                        >
                          <td className="patient-id">{patient.patient_id}</td>
                          <td>{formatLabel(patient.triage_category)}</td>
                          <td>{formatLabel(patient.assessment_status)}</td>
                          <td>{patient.ambulance_id ?? "Not allocated"}</td>
                          <td>{formatLabel(patient.transport_status)}</td>
                        </tr>
                      ))}
                      {filteredPatients.length === 0 && (
                        <tr>
                          <td colSpan={5} style={{ textAlign: "center", padding: "25px" }}>
                            {patients.length === 0
                              ? "Start a simulation to load patients."
                              : "No matching patients found."}
                          </td>
                        </tr>
                      )}
                    </tbody>
                  </table>
                </div>
              </section>

              {selectedPatient && (
                <section className="panel patient-details-panel">
                  <div className="panel-heading">
                    <div>
                      <h3>{selectedPatient.patient_id}</h3>
                      <p>Selected patient status</p>
                    </div>
                    <button
                      className="text-button"
                      onClick={() => setSelectedPatientId(null)}
                    >
                      Clear selection
                    </button>
                  </div>
                  <div className="patient-details-grid">
                    <div className="detail-item">
                      <span>Triage</span>
                      <strong>{formatLabel(selectedPatient.triage_category)}</strong>
                    </div>
                    <div className="detail-item">
                      <span>Assessment</span>
                      <strong>{formatLabel(selectedPatient.assessment_status)}</strong>
                    </div>
                    <div className="detail-item">
                      <span>Ambulance</span>
                      <strong>{selectedPatient.ambulance_id ?? "Not allocated"}</strong>
                    </div>
                    <div className="detail-item">
                      <span>Transport</span>
                      <strong>{formatLabel(selectedPatient.transport_status)}</strong>
                    </div>
                  </div>
                </section>
              )}
            </>
          )}

          {activePage === "resources" && (
            <>
              <div className="page-heading">
                <div>
                  <div className="eyebrow">INCIDENT COMMAND / RESOURCES</div>
                  <h1>Resource management</h1>
                  <p>Live ambulance availability and assignments.</p>
                </div>
                <span className="record-count">
                  {ambulances.filter((ambulance) => ambulance.status === "available").length} AVAILABLE
                </span>
              </div>

              <section className="metrics-grid">
                <Metric
                  icon={<Ambulance />}
                  label="Total ambulances"
                  value={String(ambulances.length)}
                  note="Registered simulator resources"
                />
                <Metric
                  icon={<Activity />}
                  label="Available"
                  value={String(ambulances.filter((ambulance) => ambulance.status === "available").length)}
                  note="Ready for allocation"
                />
                <Metric
                  icon={<Users />}
                  label="Occupied"
                  value={String(ambulances.filter((ambulance) => ambulance.status === "occupied").length)}
                  note="Currently assigned"
                />
                <Metric
                  icon={<Clock3 />}
                  label="Simulation time"
                  value={formatClock(elapsedSeconds)}
                  note="Current elapsed time"
                />
              </section>

              <section className="panel workspace-panel">
                <div className="panel-heading">
                  <div>
                    <h3>Ambulance inventory</h3>
                    <p>Current resource assignments</p>
                  </div>
                </div>
                <div className="resource-grid">
                  {ambulances.map((ambulance) => (
                    <div className="resource-card" key={ambulance.ambulance_id}>
                      <div className="resource-card-top">
                        <Ambulance size={22} />
                        <span className={`resource-status ${ambulance.status}`}>
                          {formatLabel(ambulance.status)}
                        </span>
                      </div>
                      <h3>{ambulance.ambulance_id}</h3>
                      <p>
                        Assigned patient:{" "}
                        <strong>{ambulance.assigned_patient_id ?? "None"}</strong>
                      </p>
                    </div>
                  ))}
                  {ambulances.length === 0 && (
                    <p className="empty-state">Start a simulation to load resources.</p>
                  )}
                </div>
              </section>
            </>
          )}

          {activePage === "timeline" && (
            <>
              <div className="page-heading">
                <div>
                  <div className="eyebrow">INCIDENT COMMAND / ACTIVITY</div>
                  <h1>Event timeline</h1>
                  <p>Chronological record of simulator events.</p>
                </div>
                <span className="record-count">{events.length} EVENTS</span>
              </div>

              <section className="panel workspace-panel">
                <div className="panel-heading">
                  <div>
                    <h3>Simulation activity</h3>
                    <p>Events recorded by the simulation engine</p>
                  </div>
                </div>
                <div className="timeline-list">
                  {[...events].reverse().map((event, index) => (
                    <div
                      className="timeline-item"
                      key={`${event.simulation_second}-${index}`}
                    >
                      <div className="timeline-marker" />
                      <div className="timeline-content">
                        <div className="timeline-meta">
                          <strong>{formatLabel(event.event_type)}</strong>
                          <span>{formatClock(event.simulation_second)}</span>
                        </div>
                        <p>{event.description}</p>
                        {event.patient_id && (
                          <span className="timeline-patient">
                            Patient {event.patient_id}
                          </span>
                        )}
                      </div>
                    </div>
                  ))}
                  {events.length === 0 && (
                    <p className="empty-state">No simulator events recorded yet.</p>
                  )}
                </div>
              </section>
            </>
          )}
        </section>
      </main>
    </div>
  );
}

function Metric({
  icon,
  label,
  value,
  note,
}: {
  icon: ReactNode;
  label: string;
  value: string;
  note: string;
}) {
  return (
    <div className="metric-card">
      <div className="metric-top">
        <span>{label}</span>
        <div className="metric-icon">{icon}</div>
      </div>
      <strong className="metric-value">{value}</strong>
      <span className="metric-note">{note}</span>
    </div>
  );
}

export default App;
