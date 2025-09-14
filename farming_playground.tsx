import React, { useState, useEffect } from 'react';
import { Sprout, BarChart3, DollarSign, TrendingUp, MessageSquare, Rocket, Upload, MapPin, Thermometer, Droplets, Zap, Sun, CloudRain } from 'lucide-react';
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer, BarChart, Bar, PieChart, Pie, Cell, RadarChart, PolarGrid, PolarAngleAxis, PolarRadiusAxis, Radar } from 'recharts';

const PreciseFarmingPlayground = () => {
  const [activeTab, setActiveTab] = useState('soil');
  const [soilData, setSoilData] = useState(null);
  const [selectedCrop, setSelectedCrop] = useState('wheat');
  const [farmSize, setFarmSize] = useState(10);
  const [weatherData, setWeatherData] = useState({
    temperature: 22,
    humidity: 65,
    rainfall: 120,
    sunlight: 8
  });

  // Sample soil analysis data
  const sampleSoilData = {
    pH: 6.8,
    nitrogen: 45,
    phosphorus: 30,
    potassium: 65,
    organicMatter: 3.2,
    moisture: 28,
    temperature: 18,
    conductivity: 1.2,
    location: "Field A - North Section"
  };

  // Crop database with requirements and characteristics
  const cropDatabase = {
    wheat: {
      name: 'Wheat',
      season: 'Winter',
      pHRange: [6.0, 7.5],
      nitrogenReq: 40,
      phosphorusReq: 25,
      potassiumReq: 30,
      waterReq: 450,
      growthDays: 120,
      yieldPerHectare: 3.5,
      pricePerTon: 250,
      seedCost: 150,
      fertilizerCost: 200,
      laborCost: 300,
      icon: '🌾'
    },
    corn: {
      name: 'Corn',
      season: 'Summer',
      pHRange: [6.0, 6.8],
      nitrogenReq: 60,
      phosphorusReq: 35,
      potassiumReq: 40,
      waterReq: 600,
      growthDays: 100,
      yieldPerHectare: 8.5,
      pricePerTon: 200,
      seedCost: 200,
      fertilizerCost: 350,
      laborCost: 400,
      icon: '🌽'
    },
    rice: {
      name: 'Rice',
      season: 'Monsoon',
      pHRange: [5.5, 6.5],
      nitrogenReq: 50,
      phosphorusReq: 30,
      potassiumReq: 35,
      waterReq: 1200,
      growthDays: 140,
      yieldPerHectare: 4.5,
      pricePerTon: 300,
      seedCost: 180,
      fertilizerCost: 280,
      laborCost: 450,
      icon: '🌾'
    },
    soybean: {
      name: 'Soybean',
      season: 'Summer',
      pHRange: [6.0, 7.0],
      nitrogenReq: 25,
      phosphorusReq: 40,
      potassiumReq: 50,
      waterReq: 500,
      growthDays: 110,
      yieldPerHectare: 2.8,
      pricePerTon: 400,
      seedCost: 220,
      fertilizerCost: 180,
      laborCost: 350,
      icon: '🫘'
    }
  };

  // Historical yield data for prediction
  const yieldHistoryData = [
    { year: '2020', wheat: 3.2, corn: 7.8, rice: 4.1, soybean: 2.5 },
    { year: '2021', wheat: 3.6, corn: 8.1, rice: 4.3, soybean: 2.7 },
    { year: '2022', wheat: 3.4, corn: 8.3, rice: 4.2, soybean: 2.6 },
    { year: '2023', wheat: 3.8, corn: 8.7, rice: 4.6, soybean: 2.9 },
    { year: '2024', wheat: 3.5, corn: 8.2, rice: 4.4, soybean: 2.8 }
  ];

  const [feedback, setFeedback] = useState({
    farmerName: '',
    location: '',
    cropType: '',
    satisfaction: 5,
    comments: '',
    improvements: []
  });

  // Soil analysis component
  const SoilAnalyzer = () => {
    const handleFileUpload = () => {
      // Simulate file upload
      setSoilData(sampleSoilData);
    };

    const getSoilHealthScore = () => {
      if (!soilData) return 0;
      const pHScore = soilData.pH >= 6.0 && soilData.pH <= 7.0 ? 100 : 70;
      const nutrientScore = (soilData.nitrogen + soilData.phosphorus + soilData.potassium) / 3;
      const moistureScore = soilData.moisture >= 20 && soilData.moisture <= 35 ? 100 : 70;
      return Math.round((pHScore + nutrientScore + moistureScore) / 3);
    };

    const soilHealthData = soilData ? [
      { subject: 'pH Level', A: (soilData.pH / 14) * 100, fullMark: 100 },
      { subject: 'Nitrogen', A: soilData.nitrogen, fullMark: 100 },
      { subject: 'Phosphorus', A: soilData.phosphorus, fullMark: 100 },
      { subject: 'Potassium', A: soilData.potassium, fullMark: 100 },
      { subject: 'Organic Matter', A: soilData.organicMatter * 20, fullMark: 100 },
      { subject: 'Moisture', A: soilData.moisture, fullMark: 100 }
    ] : [];

    return (
      <div className="space-y-6">
        <div className="flex items-center gap-3">
          <Sprout className="text-green-500" size={24} />
          <h2 className="text-2xl font-bold">Soil Analysis Dashboard</h2>
        </div>

        {!soilData ? (
          <div className="bg-gradient-to-br from-green-50 to-blue-50 p-8 rounded-lg border-2 border-dashed border-green-300">
            <div className="text-center">
              <Upload className="mx-auto text-green-500 mb-4" size={48} />
              <h3 className="text-lg font-semibold mb-2">Upload Soil Data</h3>
              <p className="text-gray-600 mb-4">Upload your soil test results to get detailed analysis</p>
              <button 
                onClick={handleFileUpload}
                className="bg-green-500 hover:bg-green-600 text-white px-6 py-2 rounded-lg transition-colors"
              >
                Simulate Upload
              </button>
            </div>
          </div>
        ) : (
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
            <div className="bg-white p-6 rounded-lg shadow-lg border">
              <h3 className="font-bold text-lg mb-4 flex items-center gap-2">
                <MapPin className="text-blue-500" size={20} />
                Sample Information
              </h3>
              <div className="space-y-3">
                <div className="flex justify-between">
                  <span className="font-medium">Location:</span>
                  <span className="text-gray-700">{soilData.location}</span>
                </div>
                <div className="flex justify-between">
                  <span className="font-medium">pH Level:</span>
                  <span className={`font-semibold ${soilData.pH >= 6.0 && soilData.pH <= 7.5 ? 'text-green-600' : 'text-orange-600'}`}>
                    {soilData.pH}
                  </span>
                </div>
                <div className="flex justify-between">
                  <span className="font-medium">Nitrogen (N):</span>
                  <span className="text-gray-700">{soilData.nitrogen} ppm</span>
                </div>
                <div className="flex justify-between">
                  <span className="font-medium">Phosphorus (P):</span>
                  <span className="text-gray-700">{soilData.phosphorus} ppm</span>
                </div>
                <div className="flex justify-between">
                  <span className="font-medium">Potassium (K):</span>
                  <span className="text-gray-700">{soilData.potassium} ppm</span>
                </div>
                <div className="flex justify-between">
                  <span className="font-medium">Organic Matter:</span>
                  <span className="text-gray-700">{soilData.organicMatter}%</span>
                </div>
              </div>
            </div>

            <div className="bg-white p-6 rounded-lg shadow-lg border">
              <h3 className="font-bold text-lg mb-4">Soil Health Score</h3>
              <div className="text-center">
                <div className={`text-6xl font-bold mb-2 ${getSoilHealthScore() >= 80 ? 'text-green-500' : getSoilHealthScore() >= 60 ? 'text-yellow-500' : 'text-red-500'}`}>
                  {getSoilHealthScore()}
                </div>
                <div className="text-gray-600">Overall Health Score</div>
              </div>
            </div>

            <div className="lg:col-span-2 bg-white p-6 rounded-lg shadow-lg border">
              <h3 className="font-bold text-lg mb-4">Nutrient Profile Radar</h3>
              <ResponsiveContainer width="100%" height={300}>
                <RadarChart data={soilHealthData}>
                  <PolarGrid />
                  <PolarAngleAxis dataKey="subject" />
                  <PolarRadiusAxis />
                  <Radar
                    name="Soil Nutrients"
                    dataKey="A"
                    stroke="#10b981"
                    fill="#10b981"
                    fillOpacity={0.3}
                  />
                </RadarChart>
              </ResponsiveContainer>
            </div>
          </div>
        )}
      </div>
    );
  };

  // Crop recommendation component
  const CropRecommender = () => {
    const getRecommendations = () => {
      if (!soilData) return [];
      
      return Object.entries(cropDatabase).map(([key, crop]) => {
        let suitabilityScore = 0;
        let reasons = [];

        // pH compatibility
        if (soilData.pH >= crop.pHRange[0] && soilData.pH <= crop.pHRange[1]) {
          suitabilityScore += 25;
          reasons.push('✓ pH compatible');
        } else {
          reasons.push('⚠ pH not optimal');
        }

        // Nutrient availability
        if (soilData.nitrogen >= crop.nitrogenReq) {
          suitabilityScore += 25;
          reasons.push('✓ Sufficient nitrogen');
        } else {
          reasons.push('⚠ Low nitrogen');
        }

        if (soilData.phosphorus >= crop.phosphorusReq) {
          suitabilityScore += 25;
        }

        if (soilData.potassium >= crop.potassiumReq) {
          suitabilityScore += 25;
        }

        return {
          crop: key,
          name: crop.name,
          icon: crop.icon,
          score: suitabilityScore,
          reasons: reasons,
          season: crop.season
        };
      }).sort((a, b) => b.score - a.score);
    };

    const recommendations = getRecommendations();

    return (
      <div className="space-y-6">
        <div className="flex items-center gap-3">
          <Sprout className="text-green-500" size={24} />
          <h2 className="text-2xl font-bold">Crop Recommendation Engine</h2>
        </div>

        <div className="grid grid-cols-1 md:grid-cols-2 gap-4 mb-6">
          <div className="bg-white p-4 rounded-lg shadow border">
            <h3 className="font-semibold mb-2 flex items-center gap-2">
              <Thermometer className="text-red-500" size={16} />
              Weather Conditions
            </h3>
            <div className="space-y-2 text-sm">
              <div>Temperature: {weatherData.temperature}°C</div>
              <div>Humidity: {weatherData.humidity}%</div>
              <div>Annual Rainfall: {weatherData.rainfall}mm</div>
              <div>Daily Sunlight: {weatherData.sunlight} hours</div>
            </div>
          </div>
          <div className="bg-white p-4 rounded-lg shadow border">
            <h3 className="font-semibold mb-2">Soil Summary</h3>
            {soilData ? (
              <div className="space-y-2 text-sm">
                <div>pH: {soilData.pH}</div>
                <div>N-P-K: {soilData.nitrogen}-{soilData.phosphorus}-{soilData.potassium}</div>
                <div>Moisture: {soilData.moisture}%</div>
                <div>Organic Matter: {soilData.organicMatter}%</div>
              </div>
            ) : (
              <p className="text-gray-500">Upload soil data first</p>
            )}
          </div>
        </div>

        {recommendations.length > 0 ? (
          <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
            {recommendations.map((rec, idx) => (
              <div key={rec.crop} className={`p-6 rounded-lg shadow-lg border-2 ${
                idx === 0 ? 'border-green-500 bg-green-50' : 'border-gray-200 bg-white'
              }`}>
                <div className="flex items-center justify-between mb-3">
                  <div className="flex items-center gap-3">
                    <span className="text-2xl">{rec.icon}</span>
                    <div>
                      <h3 className="font-bold text-lg">{rec.name}</h3>
                      <span className="text-sm text-gray-600">{rec.season} Season</span>
                    </div>
                  </div>
                  <div className={`text-2xl font-bold ${
                    rec.score >= 75 ? 'text-green-500' : rec.score >= 50 ? 'text-yellow-500' : 'text-red-500'
                  }`}>
                    {rec.score}%
                  </div>
                </div>
                <div className="space-y-1">
                  {rec.reasons.map((reason, i) => (
                    <div key={i} className={`text-sm ${reason.includes('✓') ? 'text-green-600' : 'text-orange-600'}`}>
                      {reason}
                    </div>
                  ))}
                </div>
                {idx === 0 && (
                  <div className="mt-3 bg-green-100 p-2 rounded text-sm font-medium text-green-800">
                    🏆 Best Recommendation
                  </div>
                )}
              </div>
            ))}
          </div>
        ) : (
          <div className="text-center py-8 text-gray-500">
            Upload soil data to get crop recommendations
          </div>
        )}
      </div>
    );
  };

  // Economics dashboard component
  const EconomicsDashboard = () => {
    const calculateEconomics = (cropKey) => {
      const crop = cropDatabase[cropKey];
      const totalCosts = crop.seedCost + crop.fertilizerCost + crop.laborCost;
      const grossRevenue = crop.yieldPerHectare * crop.pricePerTon * farmSize;
      const totalCost = totalCosts * farmSize;
      const netProfit = grossRevenue - totalCost;
      const profitMargin = (netProfit / grossRevenue) * 100;

      return {
        grossRevenue,
        totalCost,
        netProfit,
        profitMargin,
        breakEvenPrice: totalCosts / crop.yieldPerHectare
      };
    };

    const economicsData = Object.entries(cropDatabase).map(([key, crop]) => {
      const economics = calculateEconomics(key);
      return {
        name: crop.name,
        revenue: economics.grossRevenue,
        cost: economics.totalCost,
        profit: economics.netProfit,
        margin: economics.profitMargin
      };
    });

    const costBreakdown = cropDatabase[selectedCrop] ? [
      { name: 'Seeds', value: cropDatabase[selectedCrop].seedCost * farmSize, color: '#8884d8' },
      { name: 'Fertilizer', value: cropDatabase[selectedCrop].fertilizerCost * farmSize, color: '#82ca9d' },
      { name: 'Labor', value: cropDatabase[selectedCrop].laborCost * farmSize, color: '#ffc658' }
    ] : [];

    return (
      <div className="space-y-6">
        <div className="flex items-center gap-3">
          <DollarSign className="text-green-500" size={24} />
          <h2 className="text-2xl font-bold">Farm Economics Dashboard</h2>
        </div>

        <div className="grid grid-cols-1 md:grid-cols-3 gap-4 mb-6">
          <div className="bg-white p-4 rounded-lg shadow border">
            <label className="block text-sm font-medium mb-2">Select Crop</label>
            <select 
              value={selectedCrop}
              onChange={(e) => setSelectedCrop(e.target.value)}
              className="w-full p-2 border rounded focus:ring-2 focus:ring-green-500"
            >
              {Object.entries(cropDatabase).map(([key, crop]) => (
                <option key={key} value={key}>{crop.name}</option>
              ))}
            </select>
          </div>
          <div className="bg-white p-4 rounded-lg shadow border">
            <label className="block text-sm font-medium mb-2">Farm Size (hectares)</label>
            <input 
              type="number"
              value={farmSize}
              onChange={(e) => setFarmSize(Number(e.target.value))}
              className="w-full p-2 border rounded focus:ring-2 focus:ring-green-500"
            />
          </div>
          <div className="bg-green-50 p-4 rounded-lg border border-green-200">
            <div className="text-center">
              <div className="text-2xl font-bold text-green-600">
                ${calculateEconomics(selectedCrop).netProfit.toLocaleString()}
              </div>
              <div className="text-sm text-green-700">Projected Net Profit</div>
            </div>
          </div>
        </div>

        <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
          <div className="bg-white p-6 rounded-lg shadow border">
            <h3 className="font-bold text-lg mb-4">Profit Comparison</h3>
            <ResponsiveContainer width="100%" height={300}>
              <BarChart data={economicsData}>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis dataKey="name" />
                <YAxis />
                <Tooltip formatter={(value) => `$${value.toLocaleString()}`} />
                <Legend />
                <Bar dataKey="profit" fill="#10b981" name="Net Profit" />
                <Bar dataKey="cost" fill="#ef4444" name="Total Cost" />
              </BarChart>
            </ResponsiveContainer>
          </div>

          <div className="bg-white p-6 rounded-lg shadow border">
            <h3 className="font-bold text-lg mb-4">Cost Breakdown - {cropDatabase[selectedCrop].name}</h3>
            <ResponsiveContainer width="100%" height={300}>
              <PieChart>
                <Pie
                  data={costBreakdown}
                  cx="50%"
                  cy="50%"
                  outerRadius={80}
                  dataKey="value"
                  label={(entry) => `${entry.name}: $${entry.value.toLocaleString()}`}
                >
                  {costBreakdown.map((entry, index) => (
                    <Cell key={index} fill={entry.color} />
                  ))}
                </Pie>
                <Tooltip formatter={(value) => `$${value.toLocaleString()}`} />
              </PieChart>
            </ResponsiveContainer>
          </div>
        </div>

        <div className="bg-white p-6 rounded-lg shadow border">
          <h3 className="font-bold text-lg mb-4">Financial Metrics</h3>
          <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
            {Object.entries(calculateEconomics(selectedCrop)).map(([key, value]) => (
              <div key={key} className="text-center p-4 bg-gray-50 rounded">
                <div className="text-2xl font-bold text-blue-600">
                  {typeof value === 'number' ? 
                    (key.includes('Price') ? `$${value.toFixed(2)}` : 
                     key.includes('margin') ? `${value.toFixed(1)}%` : 
                     `$${value.toLocaleString()}`) 
                    : value}
                </div>
                <div className="text-sm text-gray-600 capitalize">
                  {key.replace(/([A-Z])/g, ' $1').trim()}
                </div>
              </div>
            ))}
          </div>
        </div>
      </div>
    );
  };

  // Yield predictor component
  const YieldPredictor = () => {
    const [predictions, setPredictions] = useState(null);

    const generatePredictions = () => {
      const baseYield = cropDatabase[selectedCrop].yieldPerHectare;
      const weatherFactor = (weatherData.temperature / 25) * (weatherData.humidity / 70) * (weatherData.rainfall / 600);
      const soilFactor = soilData ? (soilData.nitrogen + soilData.phosphorus + soilData.potassium) / 150 : 0.8;
      
      const predictedYield = baseYield * weatherFactor * soilFactor;
      const confidence = Math.min(95, 70 + (soilData ? 15 : 0) + (weatherFactor > 0.8 ? 10 : 0));

      setPredictions({
        predicted: predictedYield,
        confidence: confidence,
        factors: {
          weather: weatherFactor,
          soil: soilFactor,
          baseline: baseYield
        }
      });
    };

    useEffect(() => {
      generatePredictions();
    }, [selectedCrop, soilData, weatherData]);

    const yieldTrendData = yieldHistoryData.map(item => ({
      year: item.year,
      actual: item[selectedCrop],
      predicted: item[selectedCrop] * (0.95 + Math.random() * 0.1)
    }));

    return (
      <div className="space-y-6">
        <div className="flex items-center gap-3">
          <TrendingUp className="text-blue-500" size={24} />
          <h2 className="text-2xl font-bold">AI Yield Predictor</h2>
        </div>

        <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
          <div className="bg-gradient-to-br from-blue-50 to-indigo-50 p-6 rounded-lg border border-blue-200">
            <h3 className="font-bold text-lg mb-2">Predicted Yield</h3>
            <div className="text-3xl font-bold text-blue-600 mb-2">
              {predictions ? predictions.predicted.toFixed(2) : '-.--'} t/ha
            </div>
            <div className="text-sm text-blue-700">
              For {cropDatabase[selectedCrop].name}
            </div>
          </div>
          
          <div className="bg-gradient-to-br from-green-50 to-emerald-50 p-6 rounded-lg border border-green-200">
            <h3 className="font-bold text-lg mb-2">Confidence Level</h3>
            <div className="text-3xl font-bold text-green-600 mb-2">
              {predictions ? predictions.confidence.toFixed(0) : '--'}%
            </div>
            <div className="text-sm text-green-700">
              Based on available data
            </div>
          </div>

          <div className="bg-gradient-to-br from-purple-50 to-violet-50 p-6 rounded-lg border border-purple-200">
            <h3 className="font-bold text-lg mb-2">Potential Revenue</h3>
            <div className="text-3xl font-bold text-purple-600 mb-2">
              ${predictions ? (predictions.predicted * cropDatabase[selectedCrop].pricePerTon * farmSize).toLocaleString() : '--'}
            </div>
            <div className="text-sm text-purple-700">
              Total farm revenue
            </div>
          </div>
        </div>

        <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
          <div className="bg-white p-6 rounded-lg shadow border">
            <h3 className="font-bold text-lg mb-4">Historical vs Predicted Yield</h3>
            <ResponsiveContainer width="100%" height={300}>
              <LineChart data={yieldTrendData}>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis dataKey="year" />
                <YAxis />
                <Tooltip />
                <Legend />
                <Line type="monotone" dataKey="actual" stroke="#10b981" name="Historical Yield" />
                <Line type="monotone" dataKey="predicted" stroke="#3b82f6" strokeDasharray="5 5" name="Predicted Yield" />
              </LineChart>
            </ResponsiveContainer>
          </div>

          <div className="bg-white p-6 rounded-lg shadow border">
            <h3 className="font-bold text-lg mb-4">Prediction Factors</h3>
            {predictions && (
              <div className="space-y-4">
                <div className="flex justify-between items-center">
                  <span>Weather Impact</span>
                  <div className="flex items-center gap-2">
                    <div className="w-24 h-2 bg-gray-200 rounded">
                      <div 
                        className="h-full bg-blue-500 rounded"
                        style={{ width: `${predictions.factors.weather * 100}%` }}
                      ></div>
                    </div>
                    <span className="text-sm">{(predictions.factors.weather * 100).toFixed(0)}%</span>
                  </div>
                </div>
                <div className="flex justify-between items-center">
                  <span>Soil Quality</span>
                  <div className="flex items-center gap-2">
                    <div className="w-24 h-2 bg-gray-200 rounded">
                      <div 
                        className="h-full bg-green-500 rounded"
                        style={{ width: `${predictions.factors.soil * 100}%` }}
                      ></div>
                    </div>
                    <span className="text-sm">{(predictions.factors.soil * 100).toFixed(0)}%</span>
                  </div>
                </div>
                <div className="bg-gray-50 p-3 rounded">
                  <div className="text-sm text-gray-600">
                    <strong>Model Inputs:</strong> Temperature, humidity, rainfall, soil nutrients, historical data
                  </div>
                </div>
              </div>
            )}
          </div>
        </div>
      </div>
    );
  };

  // Feedback form component
  const FeedbackForm = () => {
    const [submitted, setSubmitted] = useState(false);

    const handleSubmit = (e) => {
      e.preventDefault();
      setSubmitted(true);
      setTimeout(() => setSubmitted(false), 3000);
    };

    return (
      <div className="space-y-6">
        <div className="flex items-center gap-3">
          <MessageSquare className="text-green-500" size={24} />
          <h2 className="text-2xl font-bold">Farmer Feedback & Experience</h2>
        </div>

        {submitted ? (
          <div className="bg-green-50 border border-green-200 p-8 rounded-lg text-center">
            <div className="text-green-600 text-6xl mb-4">✅</div>
            <h3 className="text-xl font-bold text-green-800 mb-2">Thank You!</h3>
            <p className="text-green-700">Your feedback has been submitted successfully.</p>
          </div>
        ) : (
          <form onSubmit={handleSubmit} className="space-y-6">
            <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
              <div className="bg-white p-4 rounded-lg shadow border">
                <label className="block text-sm font-medium mb-2">Farmer Name</label>
                <input 
                  type="text"
                  value={feedback.farmerName}
                  onChange={(e) => setFeedback({...feedback, farmerName: e.target.value})}
                  className="w-full p-2 border rounded focus:ring-2 focus:ring-green-500"
                  placeholder="Enter your name"
                />
              </div>
              <div className="bg-white p-4 rounded-lg shadow border">
                <label className="block text-sm font-medium mb-2">Farm Location</label>
                <input 
                  type="text"
                  value={feedback.location}
                  onChange={(e) => setFeedback({...feedback, location: e.target.value})}
                  className="w-full p-2 border rounded focus:ring-2 focus:ring-green-500"
                  placeholder="City, State/Province"
                />
              </div>
            </div>

            <div className="bg-white p-4 rounded-lg shadow border">
              <label className="block text-sm font-medium mb-2">Primary Crop Type</label>
              <select 
                value={feedback.cropType}
                onChange={(e) => setFeedback({...feedback, cropType: e.target.value})}
                className="w-full p-2 border rounded focus:ring-2 focus:ring-green-500"
              >
                <option value="">Select your main crop</option>
                {Object.entries(cropDatabase).map(([key, crop]) => (
                  <option key={key} value={key}>{crop.name}</option>
                ))}
                <option value="other">Other</option>
              </select>
            </div>

            <div className="bg-white p-4 rounded-lg shadow border">
              <label className="block text-sm font-medium mb-2">Satisfaction with Platform (1-10)</label>
              <div className="flex items-center space-x-2">
                <input 
                  type="range"
                  min="1"
                  max="10"
                  value={feedback.satisfaction}
                  onChange={(e) => setFeedback({...feedback, satisfaction: Number(e.target.value)})}
                  className="flex-1"
                />
                <span className="text-2xl font-bold text-green-600 w-8">{feedback.satisfaction}</span>
              </div>
              <div className="flex justify-between text-xs text-gray-500 mt-1">
                <span>Poor</span>
                <span>Excellent</span>
              </div>
            </div>

            <div className="bg-white p-4 rounded-lg shadow border">
              <label className="block text-sm font-medium mb-2">Comments & Suggestions</label>
              <textarea 
                value={feedback.comments}
                onChange={(e) => setFeedback({...feedback, comments: e.target.value})}
                rows="4"
                className="w-full p-2 border rounded focus:ring-2 focus:ring-green-500"
                placeholder="Share your experience, suggestions, or challenges you've faced..."
              />
            </div>

            <div className="bg-white p-4 rounded-lg shadow border">
              <label className="block text-sm font-medium mb-2">Areas for Improvement</label>
              <div className="grid grid-cols-2 gap-2">
                {[
                  'User Interface', 'Data Accuracy', 'More Crop Types', 'Weather Integration',
                  'Mobile App', 'Offline Mode', 'Local Language', 'Training Videos'
                ].map((item) => (
                  <label key={item} className="flex items-center space-x-2">
                    <input 
                      type="checkbox"
                      checked={feedback.improvements.includes(item)}
                      onChange={(e) => {
                        if (e.target.checked) {
                          setFeedback({...feedback, improvements: [...feedback.improvements, item]});
                        } else {
                          setFeedback({...feedback, improvements: feedback.improvements.filter(i => i !== item)});
                        }
                      }}
                      className="rounded"
                    />
                    <span className="text-sm">{item}</span>
                  </label>
                ))}
              </div>
            </div>

            <button 
              type="submit"
              className="w-full bg-green-500 hover:bg-green-600 text-white py-3 px-6 rounded-lg font-semibold transition-colors"
            >
              Submit Feedback
            </button>
          </form>
        )}

        <div className="grid grid-cols-1 md:grid-cols-3 gap-4 mt-8">
          <div className="bg-blue-50 p-4 rounded-lg border border-blue-200">
            <h3 className="font-bold text-blue-800 mb-2">📊 Platform Usage</h3>
            <div className="text-2xl font-bold text-blue-600">1,247</div>
            <div className="text-sm text-blue-700">Active Farmers</div>
          </div>
          <div className="bg-green-50 p-4 rounded-lg border border-green-200">
            <h3 className="font-bold text-green-800 mb-2">🌾 Successful Harvests</h3>
            <div className="text-2xl font-bold text-green-600">89%</div>
            <div className="text-sm text-green-700">Above Expected Yield</div>
          </div>
          <div className="bg-purple-50 p-4 rounded-lg border border-purple-200">
            <h3 className="font-bold text-purple-800 mb-2">💡 Satisfaction Score</h3>
            <div className="text-2xl font-bold text-purple-600">4.6/5</div>
            <div className="text-sm text-purple-700">Average Rating</div>
          </div>
        </div>
      </div>
    );
  };

  // Future scope component
  const FutureScope = () => {
    const futureFeatures = [
      {
        title: 'Drone Integration',
        description: 'Real-time aerial crop monitoring and automated field mapping',
        status: 'In Development',
        eta: 'Q2 2025',
        icon: '🚁',
        impact: 'High'
      },
      {
        title: 'IoT Sensor Network',
        description: '24/7 soil moisture, temperature, and nutrient monitoring',
        status: 'Planned',
        eta: 'Q3 2025',
        icon: '📡',
        impact: 'High'
      },
      {
        title: 'Blockchain Traceability',
        description: 'End-to-end supply chain tracking and verification',
        status: 'Research',
        eta: 'Q4 2025',
        icon: '⛓️',
        impact: 'Medium'
      },
      {
        title: 'AI Disease Detection',
        description: 'Computer vision for early pest and disease identification',
        status: 'Prototype',
        eta: 'Q1 2025',
        icon: '🔬',
        impact: 'High'
      },
      {
        title: 'Climate Adaptation',
        description: 'Climate change resilience planning and crop adaptation',
        status: 'Planning',
        eta: 'Q4 2025',
        icon: '🌡️',
        impact: 'High'
      },
      {
        title: 'Market Integration',
        description: 'Direct farmer-to-buyer marketplace with price forecasting',
        status: 'Concept',
        eta: 'Q2 2026',
        icon: '🏪',
        impact: 'Medium'
      }
    ];

    const technologyTrends = [
      { year: '2025', ai: 65, iot: 45, drones: 30, blockchain: 20 },
      { year: '2026', ai: 80, iot: 70, drones: 55, blockchain: 35 },
      { year: '2027', ai: 90, iot: 85, drones: 75, blockchain: 50 },
      { year: '2028', ai: 95, iot: 95, drones: 90, blockchain: 70 }
    ];

    return (
      <div className="space-y-6">
        <div className="flex items-center gap-3">
          <Rocket className="text-purple-500" size={24} />
          <h2 className="text-2xl font-bold">Future of Precision Farming</h2>
        </div>

        <div className="bg-gradient-to-br from-purple-50 to-blue-50 p-6 rounded-lg border border-purple-200">
          <h3 className="font-bold text-xl mb-4 text-purple-800">Vision 2030</h3>
          <p className="text-gray-700 leading-relaxed">
            Transform agriculture through cutting-edge technology, creating sustainable, profitable, and resilient farming systems 
            that feed the world while protecting our planet. Our platform will integrate AI, IoT, drones, and blockchain to create 
            the most comprehensive precision farming ecosystem.
          </p>
        </div>

        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
          {futureFeatures.map((feature, idx) => (
            <div key={idx} className="bg-white p-6 rounded-lg shadow-lg border">
              <div className="flex items-center justify-between mb-3">
                <span className="text-2xl">{feature.icon}</span>
                <span className={`px-2 py-1 rounded text-xs font-semibold ${
                  feature.status === 'In Development' ? 'bg-green-100 text-green-800' :
                  feature.status === 'Prototype' ? 'bg-blue-100 text-blue-800' :
                  feature.status === 'Planned' ? 'bg-yellow-100 text-yellow-800' :
                  'bg-gray-100 text-gray-800'
                }`}>
                  {feature.status}
                </span>
              </div>
              <h3 className="font-bold text-lg mb-2">{feature.title}</h3>
              <p className="text-gray-600 text-sm mb-3">{feature.description}</p>
              <div className="flex justify-between items-center text-sm">
                <span className="text-gray-500">ETA: {feature.eta}</span>
                <span className={`px-2 py-1 rounded ${
                  feature.impact === 'High' ? 'bg-red-100 text-red-800' : 'bg-orange-100 text-orange-800'
                }`}>
                  {feature.impact} Impact
                </span>
              </div>
            </div>
          ))}
        </div>

        <div className="bg-white p-6 rounded-lg shadow border">
          <h3 className="font-bold text-lg mb-4">Technology Adoption Roadmap</h3>
          <ResponsiveContainer width="100%" height={300}>
            <LineChart data={technologyTrends}>
              <CartesianGrid strokeDasharray="3 3" />
              <XAxis dataKey="year" />
              <YAxis />
              <Tooltip formatter={(value) => `${value}%`} />
              <Legend />
              <Line type="monotone" dataKey="ai" stroke="#8b5cf6" name="AI & Machine Learning" />
              <Line type="monotone" dataKey="iot" stroke="#10b981" name="IoT Sensors" />
              <Line type="monotone" dataKey="drones" stroke="#3b82f6" name="Drone Technology" />
              <Line type="monotone" dataKey="blockchain" stroke="#f59e0b" name="Blockchain" />
            </LineChart>
          </ResponsiveContainer>
        </div>

        <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
          <div className="bg-green-50 p-6 rounded-lg border border-green-200">
            <h3 className="font-bold text-lg mb-4 text-green-800">🌱 Sustainability Goals</h3>
            <ul className="space-y-2 text-green-700">
              <li>• 30% reduction in water usage</li>
              <li>• 25% decrease in chemical inputs</li>
              <li>• 40% improvement in soil health</li>
              <li>• Carbon-neutral farming by 2030</li>
              <li>• 50% increase in biodiversity</li>
            </ul>
          </div>
          <div className="bg-blue-50 p-6 rounded-lg border border-blue-200">
            <h3 className="font-bold text-lg mb-4 text-blue-800">📈 Economic Impact</h3>
            <ul className="space-y-2 text-blue-700">
              <li>• 35% increase in farmer income</li>
              <li>• 20% reduction in operational costs</li>
              <li>• 95% accuracy in yield predictions</li>
              <li>• 15% improvement in market prices</li>
              <li>• 100,000+ farmers empowered</li>
            </ul>
          </div>
        </div>

        <div className="bg-gradient-to-r from-indigo-50 to-purple-50 p-6 rounded-lg border border-indigo-200">
          <h3 className="font-bold text-lg mb-3 text-indigo-800">🚀 Join the Revolution</h3>
          <p className="text-indigo-700 mb-4">
            Be part of the agricultural transformation. Early adopters get exclusive access to beta features, 
            priority support, and special pricing on premium tools.
          </p>
          <button className="bg-indigo-500 hover:bg-indigo-600 text-white px-6 py-2 rounded-lg transition-colors">
            Request Early Access
          </button>
        </div>
      </div>
    );
  };

  const tabs = [
    { id: 'soil', label: 'Soil Analyzer', icon: Sprout, component: SoilAnalyzer },
    { id: 'crops', label: 'Crop Recommender', icon: Sprout, component: CropRecommender },
    { id: 'economics', label: 'Economics', icon: DollarSign, component: EconomicsDashboard },
    { id: 'yield', label: 'Yield Predictor', icon: TrendingUp, component: YieldPredictor },
    { id: 'feedback', label: 'Feedback', icon: MessageSquare, component: FeedbackForm },
    { id: 'future', label: 'Future Scope', icon: Rocket, component: FutureScope },
  ];

  const ActiveComponent = tabs.find(tab => tab.id === activeTab)?.component || SoilAnalyzer;

  return (
    <div className="min-h-screen bg-gradient-to-br from-green-50 to-blue-50">
      <div className="max-w-7xl mx-auto p-6">
        <div className="text-center mb-8">
          <h1 className="text-4xl font-bold mb-4 bg-gradient-to-r from-green-600 via-blue-600 to-purple-600 bg-clip-text text-transparent">
            🌾 Precision Farming Playground
          </h1>
          <p className="text-gray-600 text-lg max-w-3xl mx-auto">
            Empowering farmers with data-driven insights for sustainable agriculture and maximum yields
          </p>
        </div>
        
        <div className="flex flex-wrap gap-2 mb-8 justify-center">
          {tabs.map(tab => (
            <button
              key={tab.id}
              onClick={() => setActiveTab(tab.id)}
              className={`px-4 py-2 rounded-lg flex items-center gap-2 transition-all transform hover:scale-105 ${
                activeTab === tab.id 
                  ? 'bg-green-500 text-white shadow-lg' 
                  : 'bg-white text-gray-700 hover:bg-green-50 shadow'
              }`}
            >
              <tab.icon size={16} />
              {tab.label}
            </button>
          ))}
        </div>

        <div className="bg-white rounded-xl shadow-xl p-8 border">
          <ActiveComponent />
        </div>
        
        <div className="mt-8 text-center text-gray-500 text-sm">
          <p>Precision Farming Playground - Transforming Agriculture Through Technology</p>
        </div>
      </div>
    </div>
  );
};

export default PreciseFarmingPlayground;