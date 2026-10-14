import React, { useEffect, useMemo, useRef, useState } from 'react';
import { MapContainer, TileLayer, Marker, Polyline, Popup, useMap } from 'react-leaflet';
import L from 'leaflet';
import { AreaChart, Area, LineChart, Line, BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, Legend } from 'recharts';
import { Activity, BatteryCharging, BatteryFull, CarFront, ChartNoAxesCombined, CircleHelp, CloudSun, Gauge, Home, Lightbulb, MapPin, Menu, Navigation, PlugZap, Route, Settings, ShieldCheck, Zap, RefreshCw, Search, AlertTriangle, CheckCircle2, ExternalLink, Thermometer, Clock, ArrowRight, Cpu, Database, Info } from 'lucide-react';
import { formatRouteDuration, recommendPotentialChargeStops, resolveEpaVehicleSelection, sampleRoutePoints } from './tripPlanner.js';
import { NASA_CYCLE_INDEX_BOUNDS, requestNasaCapacityRetention } from './nasaResearch.js';
import { fetchJson } from './apiClient.js';
import { buildDrivingRecommendations, buildPlanningConsumptionProfile, calculateWeatherAdjustedConsumption, conservativeWeatherFactor, estimatePhysicsRange, WEATHER_ADJUSTMENT_DESCRIPTION } from './planning.js';
import { JourneyRecommendations } from './JourneyRecommendations.jsx';
import BatterySOHPredictor from './BatterySOHPredictor.jsx';
import ChargingStationsView from './ChargingStationsView.jsx';
import ModelPerformanceView from './ModelPerformanceView.jsx';
import { buildDashboardWeatherEvCareAdvice } from './weatherAdvice.js';
import TripSummary from './TripSummary.jsx';
import {
  validateBatteryCapacity,
  validateSOC,
  validateConsumption,
  validateSOH,
  validateRouteLocations,
  validateAllPlanningInputs
} from './inputValidation.js';

const API = import.meta.env.VITE_API_URL || 'http://127.0.0.1:8000';
const nav = [
  {id:'dashboard',label:'Dashboard',icon:Home}, {id:'trip',label:'Trip Planner',icon:Route},
  {id:'battery',label:'Battery Health',icon:BatteryFull}, {id:'models',label:'Model Performance',icon:ChartNoAxesCombined},
  {id:'charging',label:'Charging Stations',icon:PlugZap}, {id:'tips',label:'Driving Tips',icon:Lightbulb}
];
const hyderabad = [17.4474, 78.3488];
const fmt = (n,d=1) => n===null || n===undefined || n==='' || !Number.isFinite(Number(n)) ? '—' : Number(n).toFixed(d);
function GaugeRing({value=null,displayValue=null,label,unit='',color='#059669',size=112}) { const valid=value!==null&&value!==undefined&&value!==''&&Number.isFinite(Number(value)); const safe=valid?Math.max(0,Math.min(100,Number(value))):0; const shown=displayValue!==null&&displayValue!==undefined?`${fmt(displayValue,1)}${unit}`:valid?`${fmt(safe,0)}${unit}`:'—'; const r=42,c=2*Math.PI*r; return <div className="ring-wrap" style={{width:size}}><svg width={size} height={size} viewBox="0 0 100 100"><circle cx="50" cy="50" r={r} fill="none" stroke="#E2E8F0" strokeWidth="8"/><circle cx="50" cy="50" r={r} fill="none" stroke={color} strokeWidth="8" strokeLinecap="round" strokeDasharray={`${c*safe/100} ${c}`} transform="rotate(-90 50 50)"/><text x="50" y="47" textAnchor="middle" className="ring-number" fill="#172033">{shown}</text><text x="50" y="63" textAnchor="middle" className="ring-caption" fill="#64748B">{label}</text></svg></div> }
function Stat({icon:Icon,label,value,unit,sub,color='blue',isPrimary=false}) { return <div className={`stat-card ${isPrimary?'primary-stat-card':''}`}>{isPrimary&&<span className="stat-focal-badge">PRIMARY METRIC</span>}<div className={`stat-icon ${color}`}><Icon size={19}/></div><div className="stat-copy"><span>{label}</span><strong>{value}<small>{unit}</small></strong><em>{sub}</em></div></div> }
function MapRecenter({center,points=[]}) { const map=useMap(); const pointsKey=points.map(point=>point.join(',')).join(';'); useEffect(()=>{if(points.length>1)map.fitBounds(points,{padding:[24,24]});else if(Number.isFinite(center?.[0])&&Number.isFinite(center?.[1]))map.setView(center,12);},[center,map,pointsKey]); return null; }
function MapPanel({center=hyderabad,route=null,routePoints=[],stations=[],sourceMarker=null,height=340,onStationSelect=()=>{}}) {
 const pin=new L.Icon({iconUrl:'https://unpkg.com/leaflet@1.9.4/dist/images/marker-icon.png',iconRetinaUrl:'https://unpkg.com/leaflet@1.9.4/dist/images/marker-icon-2x.png',shadowUrl:'https://unpkg.com/leaflet@1.9.4/dist/images/marker-shadow.png',iconSize:[25,41],iconAnchor:[12,41]});
 const realStationPin=L.divIcon({className:'',html:'<span style="display:grid;place-items:center;width:30px;height:30px;border:2px solid white;border-radius:50%;background:#18a875;color:white;font-weight:800;box-shadow:0 2px 7px #17243a66">🔌</span>',iconSize:[30,30],iconAnchor:[15,15]});
 const simStationPin=L.divIcon({className:'',html:'<span style="display:grid;place-items:center;width:32px;height:32px;border:2px solid white;border-radius:50%;background:#7c3aed;color:white;font-weight:800;font-size:15px;box-shadow:0 3px 8px rgba(124,58,237,0.45)">⚡</span>',iconSize:[32,32],iconAnchor:[16,16]});
 const sourcePin=L.divIcon({className:'',html:'<span style="display:grid;place-items:center;width:34px;height:34px;border:2px solid white;border-radius:50%;background:#2563eb;color:white;font-weight:800;font-size:16px;box-shadow:0 3px 8px rgba(37,99,235,0.45)">📍</span>',iconSize:[34,34],iconAnchor:[17,17]});
 const routePts=Array.isArray(route?.points)&&route.points.length?route.points:routePoints;
 const stationPts=stations.map(s=>[s.lat,s.lon]);
 const srcPt=sourceMarker&&Number.isFinite(sourceMarker.latitude)&&Number.isFinite(sourceMarker.longitude)?[sourceMarker.latitude,sourceMarker.longitude]:null;
 const boundsPts=routePts.length>1?routePts:(srcPt?[srcPt,...stationPts]:stationPts);
 const hasRoute=Boolean(route||(routePts&&routePts.length>1));
 return <div className="map-frame" style={{height}}><MapContainer center={center} zoom={12} scrollWheelZoom style={{height:'100%',width:'100%'}}><MapRecenter center={center} points={boundsPts}/><TileLayer attribution='&copy; OpenStreetMap contributors' url="https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png"/>{routePts.length>1&&<><Polyline positions={routePts} pathOptions={{color:'#3478f6',weight:5,opacity:.85}}/><Marker position={routePts[0]} icon={pin}><Popup>Trip start</Popup></Marker><Marker position={routePts[routePts.length-1]} icon={pin}><Popup>Destination</Popup></Marker></>}{srcPt&&<Marker position={srcPt} icon={sourcePin}><Popup><b>📍 Source Location</b><br/>{sourceMarker.label||'Selected Center'}</Popup></Marker>}{stations.map((station,index)=><Marker key={`${station.osmType||''}-${station.id||index}`} position={[station.lat,station.lon]} icon={station.isSimulated?simStationPin:realStationPin}><Popup><b>{station.name||'Unnamed charging station'}</b>{station.isSimulated&&<span style={{display:'inline-block',marginLeft:4,fontSize:'10px',background:'#ede9fe',color:'#6d28d9',padding:'1px 4px',borderRadius:3}}>Simulated</span>}<br/>{station.operator||'Operator not listed'}<br/>{Number.isFinite(station.distanceFromSourceKm)?`Approx. ${fmt(station.distanceFromSourceKm,1)} km from source`:Number.isFinite(station.distanceFromRouteKm)?`Approx. ${fmt(station.distanceFromRouteKm,1)} km from ${hasRoute?'route':'search center'}`:''}{station.connectors&&Object.keys(station.connectors).length?<><br/>Connectors: {Object.entries(station.connectors).map(([name,count])=>`${name} (${count})`).join(', ')}</>:null}{station.chargingPower?<><br/>Mapped output: {station.chargingPower}</>:null}{station.recommendationBadge?<><br/><b>Status:</b> {station.recommendationBadge}</>:null}<br/><button type="button" onClick={()=>onStationSelect(station)}>Inspect station details</button></Popup></Marker>)}</MapContainer></div>
}
function EPAVehicleSelector({onSelect}) {
 const [search,setSearch]=useState('');
 const [vehicles,setVehicles]=useState([]);
 const vehiclesRef=useRef([]);
 const [error,setError]=useState('');
 const [loading,setLoading]=useState(false);
 useEffect(()=>{
  const query=search.trim();
  if(query.length<2){vehiclesRef.current=[];setVehicles([]);setError('');setLoading(false);return;}
  const controller=new AbortController();
  const timer=setTimeout(async()=>{
   setLoading(true);
   try{
    const response=await fetch(`${API}/api/vehicles/epa?search=${encodeURIComponent(query)}`,{signal:controller.signal});
    const data=await response.json();
    if(!response.ok)throw Error(data.detail||'EPA vehicle catalogue unavailable');
    const results=Array.isArray(data.results)?data.results:[];
    vehiclesRef.current=results;
    setVehicles(results);
    setError('');
   }catch(e){if(e.name!=='AbortError'){vehiclesRef.current=[];setVehicles([]);setError(e.message);}}
   finally{if(!controller.signal.aborted)setLoading(false);}
  },250);
  return()=>{clearTimeout(timer);controller.abort();};
 },[search]);
 function choose(value){
  const selected=resolveEpaVehicleSelection(value,vehiclesRef.current);
  setSearch(selected?`${selected.label} · record ${selected.source_record_id}`:value);
  onSelect(selected);
 }
 return <><label className="field-label" htmlFor="epa-vehicle-search">EPA vehicle configuration</label><div className="input-with-icon"><CarFront size={17}/><input id="epa-vehicle-search" list="epa-vehicle-options" value={search} onChange={e=>choose(e.target.value)} placeholder="Search year, make, or model"/></div><datalist id="epa-vehicle-options">{vehicles.map(v=><option key={v.source_record_id} value={`${v.label} · record ${v.source_record_id}`}/>)}</datalist>{loading&&<small>Searching EPA vehicle records…</small>}{error&&<div className="error-box"><AlertTriangle size={16}/>{error}</div>}</>;
}
function App(){
 const [rangePrediction,setRangePrediction]=useState(null);
 const [rangeApiStatus,setRangeApiStatus]=useState('checking'); const [rangeApiError,setRangeApiError]=useState('');
 const [epaVehicle,setEpaVehicle]=useState(null); const [epaPrediction,setEpaPrediction]=useState(null); const [epaError,setEpaError]=useState(''); const [modelStatus,setModelStatus]=useState(null); const tripSequence=useRef(0);
 const [page,setPage]=useState('dashboard'); const [menuOpen,setMenuOpen]=useState(false); const [apiStatus,setApiStatus]=useState('checking'); const [apiData,setApiData]=useState(null); const [metrics,setMetrics]=useState(null); const [metricsError,setMetricsError]=useState(''); const [refreshing,setRefreshing]=useState(false);
 const [soh,setSoh]=useState(92); const [soc,setSoc]=useState(82); const [capacity,setCapacity]=useState(60); const [consumption,setConsumption]=useState(165);
 const [temperature,setTemperature]=useState(null); const [weather,setWeather]=useState(null); const [weatherState,setWeatherState]=useState('loading'); const [weatherError,setWeatherError]=useState(''); const [weatherEnabled,setWeatherEnabled]=useState(true); const weatherSequence=useRef(0); const weatherController=useRef(null);
 const [start,setStart]=useState('VNR VJIET, Hyderabad'); const [destination,setDestination]=useState('Charminar, Hyderabad'); const [route,setRoute]=useState(null); const [routeError,setRouteError]=useState(''); const [busy,setBusy]=useState(false);
 const [stations,setStations]=useState([]); const [stationLoading,setStationLoading]=useState(false); const [stationError,setStationError]=useState(''); const [stationState,setStationState]=useState('idle'); const [stationMeta,setStationMeta]=useState(null); const [selectedStation,setSelectedStation]=useState(null); const [stationRadius,setStationRadius]=useState(4500); const [stationLatitude,setStationLatitude]=useState(hyderabad[0]); const [stationLongitude,setStationLongitude]=useState(hyderabad[1]); const stationRequest=useRef(false);
 const [nasaCycleIndex,setNasaCycleIndex]=useState(180); const [sohPrediction,setSohPrediction]=useState(null); const [nasaLoading,setNasaLoading]=useState(false); const [toast,setToast]=useState('');
 const [testTripModel,setTestTripModel]=useState('XGBoost'); const [testTripSpeed,setTestTripSpeed]=useState(35); const [testTripPower,setTestTripPower]=useState(10); const [testTripDist,setTestTripDist]=useState(0.28); const [tripDistPrediction,setTripDistPrediction]=useState(null); const [tripDistLoading,setTripDistLoading]=useState(false); const [tripDistError,setTripDistError]=useState('');
 const [weatherSimEnabled, setWeatherSimEnabled] = useState(false);
 const [simulatedTemperature, setSimulatedTemperature] = useState(35);
 const currentWeatherAvailable=weatherState==='live'&&weather&&Number.isFinite(Number(weather.temperature));
 const dashboardWeatherAdvice = useMemo(() => buildDashboardWeatherEvCareAdvice(weather, weatherState), [weather, weatherState]);
 const weatherFactor = weatherEnabled && currentWeatherAvailable ? conservativeWeatherFactor(weather.temperature) : 1;
 const calculationWeather = useMemo(() => {
   if (weatherSimEnabled && Number.isFinite(Number(simulatedTemperature))) {
     return {
       ...(weather || { wind: 0, precipitation: 0, humidity: 50 }),
       temperature: Number(simulatedTemperature),
       isSimulated: true
     };
   }
   return weather;
 }, [weatherSimEnabled, simulatedTemperature, weather]);
 const isCalculationWeatherActive = weatherEnabled && (weatherSimEnabled ? Number.isFinite(Number(simulatedTemperature)) : currentWeatherAvailable);
 const weatherConsumptionInfo = useMemo(() => {
   return calculateWeatherAdjustedConsumption(consumption, calculationWeather, isCalculationWeatherActive);
 }, [consumption, calculationWeather, isCalculationWeatherActive]);
 const effectiveConsumption = weatherConsumptionInfo.adjustedConsumptionWhKm;
 const planningValidation = useMemo(() => {
   return validateAllPlanningInputs({
     start,
     destination,
     capacity,
     soc,
     soh,
     consumption
   });
 }, [start, destination, capacity, soc, soh, consumption]);
 const isVehicleInputValid = !validateBatteryCapacity(capacity) && !validateSOC(soc) && !validateSOH(soh) && !validateConsumption(consumption);
 const localRange = useMemo(() => {
   if (!isVehicleInputValid) return { estimatedRangeKm: 0, usableEnergyKwh: 0, unadjustedRangeKm: 0 };
   return estimatePhysicsRange({
     batteryCapacityKwh: Number(capacity),
     socPercent: Number(soc),
     planningHealthPercent: Number(soh),
     consumptionWhKm: Number(effectiveConsumption),
     weatherFactor: 1
   });
 }, [capacity, soc, soh, effectiveConsumption, isVehicleInputValid]);
 const rangeRequest = useMemo(() => {
   if (!isVehicleInputValid) return null;
   return {
     battery_capacity_kwh: Number(capacity),
     soc_percent: Number(soc),
     soh_percent: Number(soh),
     consumption_wh_km: Number(effectiveConsumption),
     weather_factor: 1.0
   };
 }, [capacity, soc, soh, effectiveConsumption, isVehicleInputValid]);
 const rangeRequestKey = rangeRequest ? JSON.stringify(rangeRequest) : null;
 const rangeKm = rangePrediction?.requestKey === rangeRequestKey ? rangePrediction.value : localRange.estimatedRangeKm;
 const remaining = route ? route.availableRangeKm - route.distanceKm : null;
 const epaBaselineWhKm = Number.isFinite(Number(epaVehicle?.features?.combined_e_kwh_100_miles)) ? Number(epaVehicle.features.combined_e_kwh_100_miles) * 1000 / 160.9344 : null;
 const consumptionProfile = useMemo(() => {
   const validCons = Number(effectiveConsumption);
   return buildPlanningConsumptionProfile(Number.isFinite(validCons) && validCons > 0 ? validCons : 165);
 }, [effectiveConsumption]);
 const filteredStations = stations.filter(item => Number.isFinite(item.distanceFromRouteKm) && item.distanceFromRouteKm <= stationRadius / 1000);
 const potentialChargeStops = route ? recommendPotentialChargeStops(filteredStations, route.availableRangeKm) : [];
 const recommendations = buildDrivingRecommendations({
   socPercent: Number(soc) || 0,
   tripDistanceKm: route?.distanceKm,
   estimatedRangeKm: route ? route.availableRangeKm : undefined,
   stationStatus: stationState === 'unavailable' ? 'unavailable' : stationState === 'empty' ? 'empty' : undefined,
   stationCount: filteredStations.length,
   ambientTemperatureC: currentWeatherAvailable ? weather.temperature : undefined,
   energyConsumptionWhKm: Number(consumption) || 165,
   epaBaselineWhKm
 });

 useEffect(() => {
  if (!rangeRequestKey) {
   setRangePrediction(null);
   return;
  }
  const controller = new AbortController();
  let active = true;
  const timer = setTimeout(async () => {
   setRangeApiStatus('loading');
   setRangeApiError('');
   try {
    const data = await fetchJson(`${API}/api/predict/range`, { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: rangeRequestKey, signal: controller.signal }, 12000);
    if (!Number.isFinite(data.estimated_range_km) || !Number.isFinite(data.usable_energy_kwh)) throw Error('Range service returned incomplete measurements.');
    if (active) { setRangePrediction({ requestKey: rangeRequestKey, value: data.estimated_range_km, usableEnergyKwh: data.usable_energy_kwh }); setRangeApiStatus('available'); }
   } catch (error) {
    if (active && error.message !== 'The request was cancelled.') { setRangePrediction(null); setRangeApiStatus('fallback'); setRangeApiError(error.message); }
   }
  }, 220);
  return () => { active = false; clearTimeout(timer); controller.abort(); };
 }, [rangeRequestKey]);

 useEffect(()=>{
  tripSequence.current+=1;
  if(busy){setBusy(false);setRouteError('Planning inputs changed. Calculate the route again.');}
  if(route)setRoute(null);
  setStations([]);setStationMeta(null);setSelectedStation(null);setStationState('idle');setStationError('');
 },[start,destination,soc,soh,capacity,consumption,weatherEnabled,epaVehicle]);

 useEffect(()=>{refreshApplicationData();},[]);
 useEffect(()=>{if(!toast)return;const t=setTimeout(()=>setToast(''),3500);return()=>clearTimeout(t)},[toast]);

 async function getWeather(lat=hyderabad[0],lon=hyderabad[1]){
  weatherSequence.current+=1; const sequence=weatherSequence.current;
  if(weatherController.current)weatherController.current.abort();
  const controller=new AbortController(); weatherController.current=controller;
  setWeatherState('loading');setWeatherError('');
  try{
   const d=await fetchJson(`${API}/api/weather/current?latitude=${encodeURIComponent(lat)}&longitude=${encodeURIComponent(lon)}`,{signal:controller.signal},12000);
   const values=[d.temperature_c,d.relative_humidity_percent,d.wind_speed_kmh,d.precipitation_mm].map(Number);
   if(values.some(value=>!Number.isFinite(value))||values[1]<0||values[1]>100||values[2]<0||values[3]<0||typeof d.observed_at!=='string')throw Error('Weather provider returned invalid current conditions.');
   const w={temperature:values[0],humidity:values[1],wind:values[2],precipitation:values[3],time:d.observed_at,timezone:d.timezone,source:d.source,location:`${Number(lat).toFixed(3)}, ${Number(lon).toFixed(3)}`,receivedAt:new Date().toISOString()};
   if(sequence===weatherSequence.current){setWeather(w);setTemperature(w.temperature);setWeatherState('live');setWeatherError('');}
   return w;
  }catch(error){
   if(sequence===weatherSequence.current&&error.message!=='The request was cancelled.'){
    setWeatherState(weather?'stale':'unavailable');setWeatherError(error.message);if(!weather)setTemperature(null);
   }
   return null;
  }
 }

 async function refreshApplicationData(){
  setRefreshing(true);setApiStatus('checking');
  const health=fetchJson(`${API}/api/health`,{},12000).then(data=>{setApiData(data);setApiStatus('online');}).catch(()=>setApiStatus('offline'));
  const model=fetchJson(`${API}/api/models/status`,{},12000).then(data=>setModelStatus(data.models)).catch(()=>setModelStatus(null));
  const score=fetchJson(`${API}/api/model-performance`,{},12000).then(data=>{setMetrics(data);setMetricsError('');}).catch(error=>setMetricsError(error.message));
  await Promise.all([health,model,score,getWeather()]);setRefreshing(false);
 }

 function selectEpaVehicle(vehicle){
  tripSequence.current+=1;setEpaVehicle(vehicle);setEpaPrediction(null);setEpaError('');setRoute(null);setRouteError('');setStations([]);setStationMeta(null);setStationState('idle');setSelectedStation(null);
  const combined=Number(vehicle?.features?.combined_e_kwh_100_miles);
  if(vehicle&&Number.isFinite(combined)&&combined>0)setConsumption(Math.round(combined*1000/160.9344));
 }

 async function loadStations(routeOverride=null){
  if(stationRequest.current||stationLoading)return;
  stationRequest.current=true;setStationLoading(true);setStationError('');setStationState('loading');setSelectedStation(null);
  try{
   const activeRoute=routeOverride||route;
   const samples=activeRoute?.points?.length>1?sampleRoutePoints(activeRoute.points,6):[[Number(stationLatitude),Number(stationLongitude)]];
   const unique=[...new Map(samples.map(point=>[point.join(','),point])).values()];
   const requestBody={points:unique.map(([latitude,longitude])=>({latitude,longitude})),route_points:activeRoute?.points?.length>1?sampleRoutePoints(activeRoute.points,1000).map(([latitude,longitude])=>({latitude,longitude})):[],radius_m:stationRadius};
   const data=await fetchJson(`${API}/api/charging-stations/near-route`,{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(requestBody)},25000);
   if(!Array.isArray(data.results))throw Error('Station provider returned an invalid results list.');
   const list=data.results.filter(x=>Number.isFinite(Number(x.latitude))&&Number.isFinite(Number(x.longitude))&&Number.isFinite(Number(x.distance_from_route_m))).map(x=>({id:x.id,osmType:x.osm_type,lat:Number(x.latitude),lon:Number(x.longitude),name:x.name,operator:x.operator||'',connectors:x.connectors||{},chargingPower:x.charging_power||null,access:x.access||null,openingHours:x.opening_hours||null,fee:x.fee||null,distanceFromRouteKm:Number(x.distance_from_route_m)/1000,routeProgressKm:Number(x.route_progress_m||0)/1000,distanceFromStartKm:Number(x.distance_from_start_m||0)/1000,distanceFromDestinationKm:Number(x.distance_from_destination_m||0)/1000,mapUrl:x.map_url})).sort((a,b)=>a.distanceFromRouteKm-b.distanceFromRouteKm);
   setStations(list);setStationMeta({source:data.source,retrievedAt:data.retrieved_at,cached:data.cached===true,dataStatus:data.data_status,notice:data.notice});setStationState(list.length?'available':'empty');
   setStationError('');
  }catch(error){setStationState(stations.length?'available':'unavailable');setStationError(error.message||'Charging-station lookup failed.');}
  finally{stationRequest.current=false;setStationLoading(false);}
 }

 async function planRoute(e){
  if(e && typeof e.preventDefault === 'function') e.preventDefault();
  if(busy)return;
  const validation = validateAllPlanningInputs({ start, destination, capacity, soc, soh, consumption });
  if(!validation.isValid) {
    const firstErr = Object.values(validation.errors)[0];
    setRouteError(firstErr);
    return;
  }
  const requestId=++tripSequence.current;setBusy(true);setRouteError('');setRoute(null);setStations([]);setStationState('idle');setStationMeta(null);setStationError('');setSelectedStation(null);setEpaError('');setEpaPrediction(null);
  try{
   const cleanStart = String(start || '').trim();
   const cleanDest = String(destination || '').trim();
   if(!cleanStart || !cleanDest) throw Error('Enter both a starting point and a destination.');
   if(cleanStart.toLowerCase() === cleanDest.toLowerCase()) throw Error('Choose different starting and destination locations.');
   const routeData=await fetchJson(`${API}/api/route`,{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({start_address:start,destination_address:destination})},45000);
   if(!Array.isArray(routeData.geometry)||routeData.geometry.length<2||!Number.isFinite(routeData.distance_m)||routeData.distance_m<=0||!Number.isFinite(routeData.duration_s)||routeData.duration_s<0)throw Error('Routing provider returned incomplete route distance, duration, or geometry.');
   const points=routeData.geometry.map(point=>[Number(point.latitude),Number(point.longitude)]);
   if(points.some(point=>point.length!==2||!point.every(Number.isFinite)))throw Error('Routing provider returned invalid route coordinates.');
   if(requestId!==tripSequence.current)return;
   let epaResult=null;
   if(epaVehicle&&Number.isSafeInteger(Number(epaVehicle.source_record_id))){
    try{
     epaResult=await fetchJson(`${API}/api/predict/epa-range/vehicle`,{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({source_record_id:Number(epaVehicle.source_record_id)})},15000);
     setEpaPrediction(epaResult);
    }catch(error){
     setEpaError(`EPA standardized estimate unavailable: ${error.message}`);
    }
   }
   let liveWeather = null;
   let destWeather = null;
   if (weatherEnabled) {
     liveWeather = await getWeather(routeData.start.latitude, routeData.start.longitude);
     if (Number.isFinite(routeData.destination?.latitude) && Number.isFinite(routeData.destination?.longitude)) {
       try {
         const destData = await fetchJson(`${API}/api/weather/current?latitude=${encodeURIComponent(routeData.destination.latitude)}&longitude=${encodeURIComponent(routeData.destination.longitude)}`, {}, 8000);
         const dVals = [destData.temperature_c, destData.relative_humidity_percent, destData.wind_speed_kmh, destData.precipitation_mm].map(Number);
         if (dVals.every(Number.isFinite) && typeof destData.observed_at === 'string') {
           destWeather = {
             temperature: dVals[0],
             humidity: dVals[1],
             wind: dVals[2],
             precipitation: dVals[3],
             time: destData.observed_at,
             source: destData.source,
             location: routeData.destination.label || `${Number(routeData.destination.latitude).toFixed(3)}, ${Number(routeData.destination.longitude).toFixed(3)}`
           };
         }
       } catch {
         destWeather = null;
       }
     }
   }
   if(requestId!==tripSequence.current)return;
   const tripWeatherFactor=weatherEnabled&&liveWeather?conservativeWeatherFactor(liveWeather.temperature):1;
   const numCap=Number(capacity);
   const numSoc=Number(soc);
   const numSoh=Number(soh);
   const numCons=Number(consumption);
   const physicsInputs={batteryCapacityKwh:numCap,socPercent:numSoc,planningHealthPercent:numSoh,consumptionWhKm:numCons,weatherFactor:tripWeatherFactor};
   let physics=estimatePhysicsRange(physicsInputs);let rangeSource='local physics calculation';
   try{
    const rangeData=await fetchJson(`${API}/api/predict/range`,{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({battery_capacity_kwh:numCap,soc_percent:numSoc,soh_percent:numSoh,consumption_wh_km:numCons,weather_factor:tripWeatherFactor})},12000);
    if(!Number.isFinite(rangeData.estimated_range_km)||!Number.isFinite(rangeData.usable_energy_kwh))throw Error('Range service returned incomplete values.');
    physics={...physics,usableEnergyKwh:rangeData.usable_energy_kwh,estimatedRangeKm:rangeData.estimated_range_km};rangeSource='FastAPI physics-based estimate';
   }catch(error){setRangeApiStatus('fallback');setRangeApiError(error.message);}
   if(requestId!==tripSequence.current)return;
   setRoute({start:[routeData.start.latitude,routeData.start.longitude],startLabel:routeData.start.label,destination:[routeData.destination.latitude,routeData.destination.longitude],destinationLabel:routeData.destination.label,points,distanceKm:routeData.distance_m/1000,durationSeconds:routeData.duration_s,epaPrediction:epaResult,availableRangeKm:physics.estimatedRangeKm,usableEnergyKwh:physics.usableEnergyKwh,unadjustedRangeKm:physics.unadjustedRangeKm,weatherFactor:tripWeatherFactor,weatherAdjusted:tripWeatherFactor!==1,rangeSource,routeSource:routeData.source,departureWeather:liveWeather,destinationWeather:destWeather});
   setRangePrediction({requestKey:rangeRequestKey,value:physics.estimatedRangeKm,usableEnergyKwh:physics.usableEnergyKwh});
   await loadStations({points,start:[routeData.start.latitude,routeData.start.longitude]});
  }catch(error){if(requestId===tripSequence.current)setRouteError(error.message||'Route calculation failed.');}
  finally{if(requestId===tripSequence.current)setBusy(false);}
 }

 async function predictSOH(e){e?.preventDefault();if(nasaLoading)return;setNasaLoading(true);setSohPrediction(null);try{const d=await requestNasaCapacityRetention({apiBase:API,cycleIndex:nasaCycleIndex,fetchImpl:(url,options)=>fetchJson(url,options,12000)});setSohPrediction(d);}catch(error){setSohPrediction({unavailable:true,message:error.message});}finally{setNasaLoading(false);}}
 async function testPredictTripDistance(e){e?.preventDefault();if(tripDistLoading)return;setTripDistLoading(true);setTripDistError('');setTripDistPrediction(null);try{const payload={model_name:testTripModel,window_distance_km:Number(testTripDist),speed_mean:Number(testTripSpeed),speed_max:Number(testTripSpeed)*1.3,speed_min:0,speed_std:Number(testTripSpeed)*0.25,power_mean:Number(testTripPower),power_max:Number(testTripPower)*2.5,power_min:-5,current_mean:Number(testTripPower)*1000/380,voltage_mean:380,soc_start:Number(soc),soc_current:Math.max(0,Number(soc)-0.2),soc_delta:-0.2,temperature_mean:Number(temperature||20)};const res=await fetchJson(`${API}/api/predict/trip-distance`,{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(payload)},12000);setTripDistPrediction(res);}catch(err){setTripDistError(err.message||'Trip distance inference failed.');}finally{setTripDistLoading(false);}}
 const performance=metrics?.experiments||metrics?.models||[];
 // EPA vehicle catalog baseline is preserved for Trip Planner and lookup; range ML comparison dropped
 const tripDistanceResearch=metrics?.trip_distance_research;
 const tripDistanceExperiments=tripDistanceResearch?.experiments||performance.filter(x=>(x.task||'').toLowerCase().includes('trip_distance'));
 const displayTripExperiments=tripDistanceExperiments.length?tripDistanceExperiments:[
  {model_name:'Training-Mean Baseline',model_type:'baseline',rmse:1.4988,mae:1.2913,r2:-0.2664,mape:131.90,architecture_notes:'Constant training-set mean (2.193 km)'},
  {model_name:'XGBoost',model_type:'gradient_boosting',rmse:1.4884,mae:1.2847,r2:-0.2491,mape:130.56,architecture_notes:'Tabular initial-window features'},
  {model_name:'LSTM',model_type:'recurrent_neural_network',rmse:1.5407,mae:1.3697,r2:-0.3392,mape:134.87,architecture_notes:'2-layer LSTM on 30s sequence'},
  {model_name:'GRU',model_type:'recurrent_neural_network',rmse:1.5843,mae:0.9052,r2:-0.4156,mape:45.60,architecture_notes:'2-layer GRU; lowest MAE (0.905 km) and MAPE (45.60%)'},
  {model_name:'Hybrid CNN-LSTM',model_type:'convolutional_recurrent_network',rmse:1.5715,mae:1.3917,r2:-0.3928,mape:141.83,architecture_notes:'1D CNN + LSTM hybrid'}
 ];
 const nasaEvaluation=metrics?.nasa_capacity_retention;
 const routeMapCenter=route?.start||[Number(stationLatitude),Number(stationLongitude)];
 const headerTitles={dashboard:['Dashboard','Your electric journey, intelligently planned'],trip:['Trip Planner','Plan a smarter trip with range and charging awareness'],battery:['Battery Health','Understand battery condition and degradation'],models:['Model Performance','Transparent evaluation of ML models'],charging:['Charging Stations','Explore mapped EV charging locations'],tips:['Helpful Driving Tips','Practical ways to improve efficiency']};
 return <div className="app-shell"><aside className={`sidebar ${menuOpen?'show':''}`}><div className="brand"><div className="brand-mark"><Zap size={22} fill="white"/></div><div><b>VoltWise</b><span>EV INTELLIGENCE</span></div><button className="mobile-close" onClick={()=>setMenuOpen(false)}>×</button></div><div className="nav-label">WORKSPACE</div><nav>{nav.map(n=><button key={n.id} className={`nav-item ${page===n.id?'active':''}`} onClick={()=>{setPage(n.id);setMenuOpen(false)}}><n.icon size={18}/><span>{n.label}</span>{n.id==='models'&&<span className="nav-pill">ML</span>}</button>)}</nav><div className="sidebar-bottom"><div className="connection"><span className={`status-dot ${apiStatus==='online'?'green':apiStatus==='offline'?'amber':''}`}/><div><b>{apiStatus==='online'?'API connected':apiStatus==='offline'?'API unavailable':'Checking API'}</b><small>{apiStatus==='online'?'FastAPI service':apiStatus==='offline'?'Backend not detected':'Connecting…'}</small></div></div><div className="profile"><div className="avatar">EV</div><div><b>EV Research Lab</b><small>Project workspace</small></div><Settings size={17} color="#64748b"/></div></div></aside><main className="main-area"><header className="topbar"><button className="hamburger" onClick={()=>setMenuOpen(!menuOpen)}><Menu/></button><div className="breadcrumbs">Workspace <span>/</span> <b>{headerTitles[page][0]}</b></div><div className="top-actions"><span className="today-chip"><span className={`status-dot ${apiStatus==='online'?'green':apiStatus==='offline'?'amber':''}`}/> {apiStatus==='online'?'System ready':apiStatus==='offline'?'API unavailable':'Checking status'}</span><button className="icon-button" title="Refresh status" onClick={refreshApplicationData} disabled={refreshing}><RefreshCw className={refreshing?'spin':''} size={17}/></button><div className="avatar small-avatar">EV</div></div></header><div className="content"><div className="page-heading"><div><div className="eyebrow">EV RANGE & BATTERY INTELLIGENCE</div><h1>{headerTitles[page][0]}</h1><p>{headerTitles[page][1]}</p></div>{page==='dashboard'&&<button className="primary-button" onClick={()=>setPage('trip')}><Route size={17}/> Plan a Trip <ArrowRight size={16}/></button>}</div>
 {page==='trip'&&<section className="panel"><div className="panel-head"><div><h3>Select an EPA vehicle configuration</h3><p>Search the official processed catalogue; this standardized estimate is not real-world range.</p></div><CarFront color="#059669"/></div><EPAVehicleSelector onSelect={selectEpaVehicle}/>{epaVehicle&&<p className="fine-print">Selected: {epaVehicle.label} · EPA record {epaVehicle.source_record_id}</p>}{epaPrediction&&<p className="fine-print">Validated EPA configuration baseline: {fmt(epaPrediction.predicted_epa_range_km,1)} km at full charge. Trip available-range estimate uses the EPA combined energy-use record with entered battery capacity, SOC ({soc}%), SOH ({soh}%), and the existing conservative weather adjustment where live weather is available. This is an estimate, not guaranteed real-world range.</p>}{route&&<p className="fine-print">The routed trip is {fmt(route.distanceKm,1)} km. {remaining<0?'The trip exceeds estimated available range; charging is needed according to this estimate.':remaining<route.availableRangeKm*.15?'A charging stop is prudent because the estimated range margin is low.':'The trip is within this planning estimate only; real-world completion is not guaranteed.'}</p>}{stationLoading&&route&&<small>Finding mapped charging stations near the route…</small>}{stationError&&route&&<div className="error-box"><AlertTriangle size={16}/>{stationError}</div>}</section>}
 {page==='battery'&&modelStatus?.battery_soh?.status==='unavailable'&&<div className="notice warning-notice"><Info size={18}/><span>{modelStatus.battery_soh.reason}</span></div>}
 {page==='dashboard'&&<>
 <div className="welcome-banner"><div className="banner-copy"><div className="banner-tag"><span className="sparkle">✦</span> INTELLIGENT MOBILITY PLATFORM</div><h2>Drive smarter. Go further.</h2><p>VoltWise delivers physics-informed range estimation, battery capacity analytics, and charging-aware route planning engineered for optimal electric travel.</p><div className="hero-actions"><button className="primary-button hero-btn" onClick={()=>setPage('trip')}><Route size={16}/> Plan a Trip <ArrowRight size={15}/></button><button className="secondary-button hero-btn" onClick={()=>setPage('battery')}><BatteryFull size={16}/> Battery Health Analytics</button></div></div><div className="banner-visual"><div className="hero-graphic"><div className="hero-glow-orb"/><div className="hero-telemetry-badge"><Zap size={18} className="telemetry-icon"/><div><span style={{fontSize:'9.5px',color:'#64748b',fontWeight:800,display:'block'}}>RANGE ENGINE</span><strong style={{fontSize:'13px',color:'#172033'}}>{fmt(rangeKm,0)} km Available</strong></div></div><div className="hero-car-silhouette"><CarFront size={90} strokeWidth={1.3}/></div></div></div></div>
 <div className="section-title-row"><div><h3>Vehicle overview</h3><p>Estimates update as you adjust the vehicle inputs.</p></div><span className="live-label"><span className={`status-dot ${rangeApiStatus==='available'?'green':'amber'}`}/> CALCULATED ESTIMATES</span></div>
 <div className="stats-grid"><Stat icon={Navigation} label="Estimated range" value={fmt(rangeKm,0)} unit=" km" sub={weatherSimEnabled ? `Physics calculation · simulated weather (${fmt(simulatedTemperature,0)}°C)` : weatherConsumptionInfo.isWeatherAdjusted ? (rangeApiStatus==='available'?'Physics calculation · weather-adjusted (API)':'Physics calculation · weather-adjusted (local)') : (rangeApiStatus==='available'?'Physics calculation (API)':'Physics calculation (local fallback)')} color="emerald" isPrimary={true}/><Stat icon={BatteryCharging} label="Battery level" value={fmt(soc,0)} unit="%" sub={`User input; ${capacity} kWh battery capacity; ${fmt(capacity*soc/100,1)} kWh nominal energy at entered SOC`} color="blue"/><Stat icon={ShieldCheck} label="Planning health adjustment" value={fmt(soh,0)} unit="%" sub="User input; not a measured battery health value" color="purple"/><Stat icon={Activity} label="Energy consumption" value={fmt(effectiveConsumption,0)} unit=" Wh/km" sub={weatherSimEnabled ? `Simulated at ${fmt(simulatedTemperature,0)}°C (base: ${consumption} Wh/km)` : weatherConsumptionInfo.isWeatherAdjusted ? `Adjusted for live weather (base: ${consumption} Wh/km)` : 'Adjustable planning input'} color="orange"/></div>
 <div className="two-col dashboard-main"><section className="panel"><div className="panel-head"><div><h3>Battery status</h3><p>Current planning inputs</p></div><span className="soft-badge">ESTIMATED</span></div><div className="gauges"><GaugeRing value={soc} label="USER SOC" unit="%" color="#059669"/><GaugeRing value={soh} label="PLAN ADJ." unit="%" color="#2563EB"/><GaugeRing value={temperature===null?null:Math.min(100,Math.max(0,temperature/60*100))} displayValue={temperature} label="AMBIENT" unit=" °C" color="#D97706"/></div><div className="input-grid"><label>State of charge <b>{soc}%</b><input type="range" min="5" max="100" value={soc} onChange={e=>setSoc(+e.target.value)}/></label><label>Planning capacity adjustment <b>{soh}%</b><input type="range" min="50" max="100" value={soh} onChange={e=>setSoh(+e.target.value)}/></label><label>Battery capacity <b>{capacity} kWh</b><input type="range" min="20" max="120" step="5" value={capacity} onChange={e=>setCapacity(+e.target.value)}/></label><label>Consumption <b>{fmt(effectiveConsumption, 0)} Wh/km{weatherConsumptionInfo.isWeatherAdjusted && <small style={{fontWeight:500,color:'var(--text-muted)',marginLeft:'5px',fontSize:'10px'}}>({weatherSimEnabled ? 'Simulated' : 'Weather-adjusted'} · base {consumption})</small>}</b><input type="range" min="100" max="300" step="5" value={consumption} onChange={e=>setConsumption(+e.target.value)}/></label></div></section><section className="panel weather-panel"><div className="panel-head"><div><h3>Local weather</h3><p>Hyderabad · Open-Meteo</p></div><CloudSun size={22} color="#D97706"/></div>{weatherSimEnabled ? <span className="soft-badge" style={{background:'#fef3c7',color:'#b45309',border:'1px solid #fcd34d'}}>SIMULATION MODE</span> : (weather&&<span className="soft-badge">{weatherState==='live'?'LIVE':'STALE'}</span>)}{weatherState==='loading'&&!weather?<div className="loading-inline"><RefreshCw className="spin" size={15}/> Loading weather...</div>:weather?<div className="weather-content"><div className="weather-temp">{fmt(weather.temperature,0)}<sup>°C</sup></div><div className="weather-details"><span><Thermometer size={15}/> Humidity {weather.humidity}%</span><span><Activity size={15}/> Wind {fmt(weather.wind,0)} km/h</span><span><CloudSun size={15}/> Rain {fmt(weather.precipitation,1)} mm</span></div></div>:<div className="empty-inline">Live weather unavailable. No live weather adjustment is applied.</div>}{weatherError&&<div className="error-box">{weatherError}</div>}<button className="text-button" onClick={()=>getWeather()} disabled={weatherState==='loading'}>{weatherState==='loading'?'Refreshing...':'Refresh weather'}</button>
<div className="weather-simulation-box" style={{marginTop:'12px',padding:'12px 14px',borderRadius:'10px',background:weatherSimEnabled?'#fffbeb':'var(--bg-surface-elevated)',border:weatherSimEnabled?'1.5px solid #f59e0b':'1px solid var(--border-subtle)',transition:'all 0.2s ease'}}>
 <div style={{display:'flex',justifyContent:'space-between',alignItems:'center',marginBottom:'8px'}}>
  <div style={{display:'flex',alignItems:'center',gap:'7px'}}>
   <Thermometer size={16} color={weatherSimEnabled?'#d97706':'var(--text-secondary)'}/>
   <strong style={{fontSize:'12.5px',color:'var(--text-primary)'}}>Weather Simulation (Testing Only)</strong>
  </div>
  <div style={{display:'flex',alignItems:'center',gap:'8px'}}>
   {weatherSimEnabled&&<span style={{fontSize:'10px',fontWeight:800,letterSpacing:'0.5px',padding:'2px 7px',borderRadius:'4px',background:'#d97706',color:'#ffffff'}}>SIMULATION MODE</span>}
   <label style={{display:'inline-flex',alignItems:'center',gap:'5px',fontSize:'11.5px',fontWeight:600,cursor:'pointer',margin:0,color:'var(--text-secondary)'}}>
    <input type="checkbox" id="weather-sim-toggle" checked={weatherSimEnabled} onChange={e=>setWeatherSimEnabled(e.target.checked)} style={{cursor:'pointer'}}/>
    <span>{weatherSimEnabled?'Simulation ON':'Simulation OFF'}</span>
   </label>
  </div>
 </div>
 <div style={{display:'grid',gridTemplateColumns:'1fr 1fr',gap:'8px',marginBottom:'10px',fontSize:'11px',padding:'6px 8px',borderRadius:'6px',background:weatherSimEnabled?'rgba(245,158,11,0.08)':'rgba(0,0,0,0.02)',border:'1px solid var(--border-subtle)'}}>
  <div>
   <span style={{display:'block',color:'var(--text-muted)',fontSize:'10px'}}>Actual API Temperature</span>
   <strong style={{color:'var(--text-primary)'}}>{weather?.temperature!==null&&weather?.temperature!==undefined?`${fmt(weather.temperature,1)} °C`:'Unavailable / Offline'}</strong>
  </div>
  <div>
   <span style={{display:'block',color:weatherSimEnabled?'#b45309':'var(--text-muted)',fontSize:'10px'}}>{weatherSimEnabled?'Simulated Temperature Used for Calculation':'Calculation Input'}</span>
   <strong style={{color:weatherSimEnabled?'#b45309':'var(--text-primary)'}}>{weatherSimEnabled?`${fmt(simulatedTemperature,1)} °C (Simulated)`:(weather?.temperature!==null&&weather?.temperature!==undefined?`${fmt(weather.temperature,1)} °C`:'Base Default (165 Wh/km)')}</strong>
  </div>
 </div>
 <div style={{display:'flex',flexDirection:'column',gap:'8px'}}>
  <div style={{display:'flex',alignItems:'center',gap:'8px',flexWrap:'wrap'}}>
   <label htmlFor="sim-temp-input" style={{fontSize:'11.5px',fontWeight:600,color:'var(--text-secondary)',margin:0}}>Test Temp (°C):</label>
   <input id="sim-temp-input" type="number" min="-30" max="55" step="1" value={simulatedTemperature} onChange={e=>{const val=e.target.value; setSimulatedTemperature(val===''? '':Number(val)); if(!weatherSimEnabled&&val!=='')setWeatherSimEnabled(true);}} placeholder="e.g. 35" style={{width:'75px',padding:'4px 8px',fontSize:'12px',borderRadius:'6px',border:'1px solid var(--border-medium)',background:'var(--bg-surface)'}}/>
   {weatherSimEnabled&&<button type="button" onClick={()=>{setWeatherSimEnabled(false); if(weather?.temperature!==null&&weather?.temperature!==undefined)setSimulatedTemperature(Math.round(weather.temperature));}} className="text-button" style={{fontSize:'11px',padding:'3px 8px',marginLeft:'auto',color:'var(--text-muted)'}} title="Disable simulation and restore live Open-Meteo temperature">Reset to Live Weather</button>}
  </div>
  <div style={{display:'flex',alignItems:'center',gap:'5px',flexWrap:'wrap'}}>
   <span style={{fontSize:'10.5px',color:'var(--text-muted)',marginRight:'2px'}}>Quick test:</span>
   {[5,25,32,35,37].map(tempVal=>(
    <button key={tempVal} type="button" onClick={()=>{setSimulatedTemperature(tempVal); setWeatherSimEnabled(true);}} style={{fontSize:'11px',fontWeight:weatherSimEnabled&&Number(simulatedTemperature)===tempVal?700:500,padding:'2px 8px',borderRadius:'5px',border:weatherSimEnabled&&Number(simulatedTemperature)===tempVal?'1.5px solid #d97706':'1px solid var(--border-subtle)',background:weatherSimEnabled&&Number(simulatedTemperature)===tempVal?'#fef3c7':'var(--bg-surface)',color:weatherSimEnabled&&Number(simulatedTemperature)===tempVal?'#92400e':'var(--text-secondary)',cursor:'pointer',transition:'all 0.15s ease'}}>{tempVal}°C</button>
   ))}
  </div>
 </div>
</div>
<p className="fine-print">{weatherSimEnabled ? `[SIMULATION ACTIVE] Calculated using simulated ${fmt(simulatedTemperature, 1)}°C (${fmt(effectiveConsumption, 1)} Wh/km). Actual live weather data is preserved.` : weatherConsumptionInfo.isWeatherAdjusted ? `${weatherConsumptionInfo.reason}. Rule-based planning heuristic, not an ML model.` : `${WEATHER_ADJUSTMENT_DESCRIPTION} ${weatherState!=='live'?'No live weather adjustment is applied.':''}`}</p>
<div className="weather-ev-care-section" style={{marginTop:'14px',paddingTop:'12px',borderTop:'1px solid var(--border-subtle)'}}>
 <div style={{display:'flex',justifyContent:'space-between',alignItems:'center',marginBottom:'8px'}}>
  <h4 style={{margin:0,fontSize:'12.5px',fontWeight:700,color:'var(--text-primary)',display:'flex',alignItems:'center',gap:'6px'}}>
   <ShieldCheck size={14} color="var(--accent-emerald)"/> Weather &amp; EV Care
  </h4>
  <span style={{fontSize:'10.5px',color:'var(--text-muted)'}}>
   {dashboardWeatherAdvice.freshnessLabel}{dashboardWeatherAdvice.observedTime?` · ${dashboardWeatherAdvice.observedTime}`:''}
  </span>
 </div>
 <div style={{display:'grid',gap:'6px'}}>
  {dashboardWeatherAdvice.items.map(item=>(
   <div key={item.id} style={{display:'flex',gap:'8px',padding:'7px 9px',borderRadius:'7px',background:item.severity==='warning'?'var(--accent-amber-dim)':item.severity==='positive'?'var(--accent-emerald-dim)':'var(--bg-surface-elevated)',border:`1px solid ${item.severity==='warning'?'var(--accent-amber-border)':item.severity==='positive'?'var(--accent-emerald-border)':'var(--border-subtle)'}`}}>
    <div style={{marginTop:'1px',flexShrink:0}}>
     {item.severity==='warning'?<AlertTriangle size={14} color="var(--accent-amber)"/>:item.severity==='positive'?<CheckCircle2 size={14} color="var(--accent-emerald)"/>:<Info size={14} color="var(--accent-blue)"/>}
    </div>
    <div style={{fontSize:'11.5px',lineHeight:'1.4'}}>
     <strong style={{display:'block',color:'var(--text-primary)',marginBottom:'1px'}}>{item.title}</strong>
     <span style={{color:'var(--text-secondary)'}}>{item.text}</span>
    </div>
   </div>
  ))}
 </div>
 <p style={{margin:'8px 0 0',fontSize:'10px',color:'var(--text-muted)',lineHeight:'1.35'}}>
  {dashboardWeatherAdvice.disclaimer}
 </p>
</div>
</section></div>
 <div className="two-col bottom-grid"><section className="panel"><div className="panel-head"><div><h3>Trip readiness</h3><p>Based on your current range estimate</p></div><Route size={19} color="#059669"/></div><div className="range-callout"><div className="range-icon"><Navigation size={20}/></div><div><span>Estimated available range</span><strong>{fmt(rangeKm,0)} km</strong></div><button className="text-button" onClick={()=>setPage('trip')}>Plan route <ArrowRight size={15}/></button></div><div className="mini-progress"><div style={{width:`${Math.min(100,soc)}%`}}/></div><div className="between"><span>{soc}% user-entered charge</span><span>{fmt(capacity*soc/100*soh/100,1)} kWh after planning adjustment</span></div>{rangeApiStatus==='fallback'&&<p className="fine-print">Range API unavailable; showing the same documented local physics calculation. {rangeApiError}</p>}</section><section className="panel"><div className="panel-head"><div><h3>Energy consumption</h3><p>Illustrative daily profile · Wh/km</p></div><span className="chart-key"><i/> Consumption</span></div><div className="chart-area"><ResponsiveContainer width="100%" height="100%"><AreaChart data={consumptionProfile} margin={{top:10,right:8,left:-20,bottom:0}}><defs><linearGradient id="energyFill" x1="0" y1="0" x2="0" y2="1"><stop offset="0%" stopColor="#2563EB" stopOpacity={0.16}/><stop offset="100%" stopColor="#2563EB" stopOpacity={0.01}/></linearGradient></defs><CartesianGrid strokeDasharray="3 3" vertical={false} stroke="#E2E8F0"/><XAxis dataKey="label" label={{value:'Illustrative time of day',position:'insideBottom',offset:-2,fill:'#64748B',fontSize:10}} axisLine={false} tickLine={false} tick={{fill:'#64748B',fontSize:10}}/><YAxis label={{value:'Energy use (Wh/km)',angle:-90,position:'insideLeft',fill:'#64748B',fontSize:10}} axisLine={false} tickLine={false} tick={{fill:'#64748B',fontSize:10}}/><Tooltip contentStyle={{backgroundColor:'#FFFFFF',borderColor:'#E2E8F0',borderRadius:'8px',boxShadow:'0 4px 16px rgba(15,23,42,0.08)',color:'#172033',fontSize:'12px'}} itemStyle={{color:'#2563EB',fontWeight:600}}/><Area type="monotone" dataKey="value" name="Wh/km" stroke="#2563EB" strokeWidth={2.5} fill="url(#energyFill)"/></AreaChart></ResponsiveContainer></div><div className="chart-foot"><span>Average shown: {fmt(consumptionProfile.reduce((s,x)=>s+x.value,0)/consumptionProfile.length,0)} Wh/km</span><span className="demo-note"><Info size={13}/> Illustrative data</span></div></section></div>
 {weather&&page==='dashboard'&&<p className="fine-print">Current conditions for {weather.location} · observed {weather.time} · {weather.source}. This weather is not an input to the EPA model.</p>}
 </>}
 {page==='trip'&&<><div className="trip-layout"><form className="panel trip-form" onSubmit={planRoute}>
 <div className="panel-head"><div><h3>Build your journey</h3><p>Route distance is retrieved from a public routing service.</p></div><Route color="#059669"/></div>
 <label className="field-label" htmlFor="trip-start-input">Starting point</label>
 <div className={`input-with-icon ${planningValidation.errors.start ? 'input-error' : ''}`}>
  <MapPin size={17}/>
  <input id="trip-start-input" value={start} onChange={e=>{setStart(e.target.value);setRoute(null);setRouteError('');}} placeholder="Enter starting point"/>
 </div>
 {planningValidation.errors.start&&<span className="field-error-text"><AlertTriangle size={13}/> {planningValidation.errors.start}</span>}
 <label className="field-label" htmlFor="trip-destination-input">Destination</label>
 <div className={`input-with-icon ${planningValidation.errors.destination ? 'input-error' : ''}`}>
  <Navigation size={17}/>
  <input id="trip-destination-input" value={destination} onChange={e=>{setDestination(e.target.value);setRoute(null);setRouteError('');}} placeholder="Enter destination"/>
 </div>
 {planningValidation.errors.destination&&<span className="field-error-text"><AlertTriangle size={13}/> {planningValidation.errors.destination}</span>}
 <div className="form-row">
  <label htmlFor="trip-capacity-input">
   Battery capacity (kWh)
   <input id="trip-capacity-input" type="number" className={planningValidation.errors.capacity ? 'input-error' : ''} value={capacity} onChange={e=>{setCapacity(e.target.value);setRoute(null);setRouteError('');}} placeholder="e.g. 60"/>
   {planningValidation.errors.capacity&&<span className="field-error-text"><AlertTriangle size={13}/> {planningValidation.errors.capacity}</span>}
  </label>
  <label htmlFor="trip-consumption-input">
   Energy use (Wh/km)
   <input id="trip-consumption-input" type="number" className={planningValidation.errors.consumption ? 'input-error' : ''} value={consumption} onChange={e=>{setConsumption(e.target.value);setRoute(null);setRouteError('');}} placeholder="e.g. 165"/>
   {planningValidation.errors.consumption&&<span className="field-error-text"><AlertTriangle size={13}/> {planningValidation.errors.consumption}</span>}
  </label>
 </div>
 <div className="form-row">
  <label htmlFor="trip-soc-input">
   Current SOC (%)
   <input id="trip-soc-input" type="number" className={planningValidation.errors.soc ? 'input-error' : ''} value={soc} onChange={e=>{setSoc(e.target.value);setRoute(null);setRouteError('');}} placeholder="0 - 100"/>
   {planningValidation.errors.soc&&<span className="field-error-text"><AlertTriangle size={13}/> {planningValidation.errors.soc}</span>}
  </label>
  <label htmlFor="trip-soh-input">
   Planning capacity adjustment (%)
   <input id="trip-soh-input" type="number" className={planningValidation.errors.soh ? 'input-error' : ''} value={soh} onChange={e=>{setSoh(e.target.value);setRoute(null);setRouteError('');}} placeholder="1 - 100"/>
   {planningValidation.errors.soh ? <span className="field-error-text"><AlertTriangle size={13}/> {planningValidation.errors.soh}</span> : <small>User-entered planning assumption; not measured SoH.</small>}
  </label>
 </div>
 <label className="toggle-row"><span><CloudSun size={17}/> Apply weather planning adjustment<small>Conservative heuristic only; not learned from a validated weather dataset.</small></span><input type="checkbox" checked={weatherEnabled} onChange={e=>setWeatherEnabled(e.target.checked)}/></label>
 <button type="submit" className="primary-button full" disabled={busy || !planningValidation.isValid}>{busy?<><RefreshCw className="spin" size={17}/> Finding route…</>:<><Search size={17}/> Calculate route &amp; range</>}</button>
 {routeError&&<div className="error-box"><AlertTriangle size={16}/>{routeError}</div>}
</form><section className="panel trip-map-panel"><div className="panel-head"><div><h3>Route map</h3><p>OpenStreetMap · OSRM routing</p></div><span className="soft-badge">{busy?'LOADING':route?'ROUTE FOUND':'READY'}</span></div><MapPanel center={routeMapCenter} route={route} routePoints={route?.points||[]} stations={stations} height={440}/>{route&&<><div className="route-metrics"><div><span>From</span><b>{route.startLabel}</b></div><div><span>To</span><b>{route.destinationLabel}</b></div><div><span>Road distance</span><b>{fmt(route.distanceKm,1)} km</b></div><div><span>Driving time</span><b>{formatRouteDuration(route.durationSeconds)}</b></div><div><span>Estimated available range</span><b>{fmt(route.availableRangeKm,0)} km</b></div></div><p className="fine-print">Road distance and driving time are returned by OSRM; station proximity is approximate. Mapped stations do not report live availability.</p></>}</section></div>{route&&<>
 <TripSummary
   route={route}
   weather={currentWeatherAvailable?weather:null}
   departureWeather={route.departureWeather||(currentWeatherAvailable?weather:null)}
   destinationWeather={route.destinationWeather||null}
   weatherState={weatherState}
   stations={filteredStations}
   soc={Number(soc)}
   capacity={Number(capacity)}
   consumption={Number(consumption)}
   onNavigateToCharging={()=>{setPage('charging');loadStations(route);}}
 />
 <JourneyRecommendations route={route} weather={currentWeatherAvailable?weather:null} departureWeather={route.departureWeather||(currentWeatherAvailable?weather:null)} destinationWeather={route.destinationWeather||null} weatherState={weatherState} soc={Number(soc)} soh={Number(soh)} capacity={Number(capacity)} consumption={Number(consumption)} epaVehicle={epaVehicle}/>
</>}</>}
 {page==='battery'&&<>
  <div className="stats-grid">
   <Stat icon={ShieldCheck} label="Trip-planning SOH input" value={fmt(soh,1)} unit="%" sub="User-set range-planning value; separate from ML prediction" color="green"/>
   <Stat icon={BatteryFull} label="Usable energy" value={fmt(capacity*soc/100*soh/100,2)} unit=" kWh" sub={`Calculated at ${capacity} kWh × ${soc}% SOC × ${soh}% planning SOH`} color="blue"/>
   <Stat icon={Thermometer} label="Ambient temperature" value={fmt(temperature,1)} unit=" °C" sub={weatherState==='live'?'Live weather context from Open-Meteo':'Weather unavailable; not applied'} color="orange"/>
  </div>
  <BatterySOHPredictor apiBase={API}/>
 </>}
 {page==='models' && (
  <ModelPerformanceView
    metrics={metrics}
    apiBase={API}
    currentSoc={soc}
    currentTemperature={temperature}
  />
 )}

 {page==='charging'&&<ChargingStationsView route={route} availableRangeKm={rangeKm} MapPanelComponent={MapPanel}/>}
 {page==='tips'&&<>
<div style={{display:'flex',justifyContent:'space-between',alignItems:'center',padding:'12px 18px',background:'var(--bg-surface)',border:'1px solid var(--border-subtle)',borderRadius:'12px',marginBottom:'16px'}}>
 <div>
  <strong style={{display:'block',fontSize:'13px',color:'var(--text-primary)'}}>Looking for real-time weather advice?</strong>
  <span style={{fontSize:'12px',color:'var(--text-secondary)'}}>Live ambient weather observations and temperature-informed EV care guidelines are available on the Dashboard.</span>
 </div>
 <button className="secondary-button" onClick={()=>setPage('dashboard')} style={{flexShrink:0,marginLeft:'16px',display:'inline-flex',alignItems:'center',gap:'6px'}}>
  View Current Weather Advice <ArrowRight size={14}/>
 </button>
</div>
<div className="tips-hero"><div><span className="banner-tag">DRIVE EFFICIENTLY</span><h2>Recommendations for your current plan</h2><p>Rule-based prompts from your entered planning values, route and available mapped station search. These are not ML recommendations.</p></div><div className="tips-hero-icon"><Lightbulb size={62}/></div></div><div className="recommendation-list">{recommendations.length?recommendations.map((item,index)=><article className={`tip-card recommendation-card priority-${item.priority||'info'}`} key={`${item.rule||item.title}-${index}`}><span className="tip-tag">{String(item.priority||'info').toUpperCase()} - RULE BASED</span><h3>{item.title}</h3><p><b>Why:</b> {item.reason}</p><p><b>Suggested action:</b> {item.action}</p>{item.trigger&&<small>Triggered by: {item.trigger}</small>}</article>):<article className="tip-card"><span className="tip-tag">CURRENT PLAN</span><h3>No additional prompt</h3><p>Available planning inputs do not currently trigger a specific recommendation. This is not a safety assessment.</p></article>}</div><h3 className="section-subheading">General driving guidance</h3><div className="tips-grid">{[{icon:Gauge,title:'Accelerate smoothly',desc:'Gentle acceleration and anticipating traffic can reduce high power demand.',tag:'ENERGY SAVING'},{icon:Thermometer,title:'Manage cabin climate',desc:'Heating and cooling consume energy. Precondition while plugged in when your vehicle supports it.',tag:'CLIMATE'},{icon:BatteryCharging,title:'Use regenerative braking',desc:'Learn the vehicle regenerative-braking settings and leave enough distance for smooth deceleration.',tag:'DRIVING'},{icon:CarFront,title:'Maintain tyre pressure',desc:'Check tyre pressure against manufacturer guidance.',tag:'MAINTENANCE'},{icon:Route,title:'Plan routes ahead',desc:'Consider elevation, traffic, road type and charger locations before long trips.',tag:'TRIP PLANNING'},{icon:ShieldCheck,title:'Follow vehicle guidance',desc:'Follow the vehicle maker charging and temperature guidance; advice varies by vehicle.',tag:'BATTERY CARE'}].map(t=><article className="tip-card" key={t.title}><div className="tip-icon"><t.icon size={21}/></div><span className="tip-tag">{t.tag}</span><h3>{t.title}</h3><p>{t.desc}</p></article>)}</div><div className="notice"><Info size={17}/><span>Ambient weather is not battery temperature. These are general recommendations, not guarantees of a particular range increase. Follow the vehicle manufacturer safety guidance.</span></div></>}
 <footer className="footer"><span>VoltWise · EV Range & Battery Intelligence</span><span>Estimates are informational and must not replace vehicle instrumentation.</span></footer></div></main>{toast&&<div className="toast"><Info size={16}/>{toast}</div>}</div>
}
export default App;
