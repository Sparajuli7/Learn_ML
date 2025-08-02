# Full-Stack Project 4: AI + IoT/Blockchain/Web3 Integration

## 🎯 Project Overview
Build a comprehensive system that integrates AI/ML with emerging technologies including IoT, Blockchain, and Web3. This project demonstrates cutting-edge integration capabilities and explores the future of decentralized, intelligent systems.

## 📋 Project Requirements

### Core Features
- **IoT Integration**: Real-time sensor data collection and AI-powered analysis
- **Blockchain Integration**: Decentralized data storage and smart contract automation
- **Web3 Integration**: Decentralized applications (dApps) with AI capabilities
- **Edge AI**: On-device AI processing for IoT devices
- **Decentralized ML**: Federated learning across distributed networks
- **Smart Contracts**: AI-powered automated decision making on blockchain

### Technical Stack
- **IoT**: Raspberry Pi + Arduino + MQTT + Node-RED
- **Blockchain**: Ethereum + Solidity + Web3.js + Hardhat
- **Web3**: MetaMask + IPFS + The Graph + Polygon
- **AI/ML**: TensorFlow Lite + ONNX + Edge Impulse
- **Infrastructure**: Docker + Kubernetes + Helm
- **Monitoring**: Grafana + Prometheus + Chainlink Oracles

---

## 🚀 Project Architecture

### 1. System Architecture Overview

```python
# integrated_system.py
from dataclasses import dataclass
from typing import Dict, List, Optional, Any
import asyncio
import logging
from datetime import datetime
from enum import Enum

class TechnologyType(Enum):
    IOT = "iot"
    BLOCKCHAIN = "blockchain"
    WEB3 = "web3"
    AI = "ai"

@dataclass
class IoTDevice:
    """IoT device configuration"""
    device_id: str
    device_type: str
    sensors: List[str]
    location: Dict[str, float]
    capabilities: List[str]
    ai_model: Optional[str]

@dataclass
class BlockchainConfig:
    """Blockchain configuration"""
    network: str  # ethereum, polygon, etc.
    contract_address: str
    gas_limit: int
    private_key: str

class IntegratedAISystem:
    def __init__(self, config: Dict):
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Initialize IoT manager
        self.iot_manager = IoTManager()
        
        # Initialize blockchain manager
        self.blockchain_manager = BlockchainManager()
        
        # Initialize Web3 manager
        self.web3_manager = Web3Manager()
        
        # Initialize AI orchestrator
        self.ai_orchestrator = AIOrchestrator()
        
        # Initialize data pipeline
        self.data_pipeline = DataPipeline()
        
        # Initialize smart contract manager
        self.smart_contract_manager = SmartContractManager()
    
    async def start_system(self):
        """Start the integrated AI system"""
        
        self.logger.info("Starting Integrated AI System...")
        
        # Start IoT devices
        await self.iot_manager.start_devices()
        
        # Deploy smart contracts
        await self.blockchain_manager.deploy_contracts()
        
        # Initialize Web3 connections
        await self.web3_manager.initialize()
        
        # Start AI processing
        await self.ai_orchestrator.start()
        
        # Start data pipeline
        await self.data_pipeline.start()
        
        self.logger.info("Integrated AI System started successfully")
    
    async def process_iot_data(self, device_id: str, sensor_data: Dict) -> Dict:
        """Process IoT sensor data with AI and blockchain integration"""
        
        try:
            # Step 1: AI Analysis
            ai_result = await self.ai_orchestrator.analyze_sensor_data(
                device_id, sensor_data
            )
            
            # Step 2: Blockchain Storage
            blockchain_result = await self.blockchain_manager.store_data(
                device_id, sensor_data, ai_result
            )
            
            # Step 3: Smart Contract Execution
            contract_result = await self.smart_contract_manager.execute_contract(
                device_id, ai_result
            )
            
            # Step 4: Web3 Integration
            web3_result = await self.web3_manager.update_dapp(
                device_id, ai_result, contract_result
            )
            
            return {
                "status": "success",
                "ai_analysis": ai_result,
                "blockchain_storage": blockchain_result,
                "smart_contract": contract_result,
                "web3_update": web3_result
            }
            
        except Exception as e:
            self.logger.error(f"Data processing failed: {str(e)}")
            return {
                "status": "failed",
                "error": str(e)
            }
```

### 2. IoT Integration System

```python
# iot_manager.py
import asyncio
import logging
from typing import Dict, List, Optional
import paho.mqtt.client as mqtt
import json
from dataclasses import dataclass

@dataclass
class SensorData:
    """Sensor data structure"""
    device_id: str
    timestamp: str
    sensor_type: str
    value: float
    unit: str
    location: Dict[str, float]

class IoTManager:
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.mqtt_client = mqtt.Client()
        self.devices: Dict[str, IoTDevice] = {}
        self.edge_ai_models = {}
        
        # Initialize MQTT
        self._setup_mqtt()
        
        # Initialize edge AI models
        self._load_edge_models()
    
    def _setup_mqtt(self):
        """Setup MQTT client for IoT communication"""
        
        self.mqtt_client.on_connect = self._on_connect
        self.mqtt_client.on_message = self._on_message
        
        # Connect to MQTT broker
        self.mqtt_client.connect("localhost", 1883, 60)
        self.mqtt_client.loop_start()
    
    def _on_connect(self, client, userdata, flags, rc):
        """MQTT connection callback"""
        self.logger.info(f"Connected to MQTT broker with result code {rc}")
        
        # Subscribe to sensor topics
        client.subscribe("sensors/+/data")
        client.subscribe("devices/+/status")
    
    def _on_message(self, client, userdata, msg):
        """MQTT message callback"""
        try:
            topic = msg.topic
            payload = json.loads(msg.payload.decode())
            
            if topic.startswith("sensors/"):
                asyncio.create_task(self._process_sensor_data(payload))
            elif topic.startswith("devices/"):
                asyncio.create_task(self._process_device_status(payload))
                
        except Exception as e:
            self.logger.error(f"Error processing MQTT message: {str(e)}")
    
    async def _process_sensor_data(self, data: Dict):
        """Process incoming sensor data"""
        
        sensor_data = SensorData(
            device_id=data["device_id"],
            timestamp=data["timestamp"],
            sensor_type=data["sensor_type"],
            value=data["value"],
            unit=data["unit"],
            location=data.get("location", {})
        )
        
        # Edge AI processing
        if sensor_data.device_id in self.edge_ai_models:
            ai_result = await self._run_edge_ai(sensor_data)
            data["ai_analysis"] = ai_result
        
        # Send to blockchain
        await self._send_to_blockchain(sensor_data)
        
        # Update Web3 dApp
        await self._update_web3_dapp(sensor_data)
    
    async def _run_edge_ai(self, sensor_data: SensorData) -> Dict:
        """Run AI analysis on edge device"""
        
        model = self.edge_ai_models.get(sensor_data.device_id)
        if not model:
            return {"status": "no_model_available"}
        
        try:
            # Preprocess data
            input_data = self._preprocess_sensor_data(sensor_data)
            
            # Run inference
            prediction = model.predict(input_data)
            
            # Post-process results
            result = self._postprocess_prediction(prediction, sensor_data)
            
            return {
                "status": "success",
                "prediction": result,
                "confidence": prediction.get("confidence", 0.0),
                "model_version": model.version
            }
            
        except Exception as e:
            self.logger.error(f"Edge AI processing failed: {str(e)}")
            return {
                "status": "failed",
                "error": str(e)
            }
    
    def _preprocess_sensor_data(self, sensor_data: SensorData) -> List[float]:
        """Preprocess sensor data for AI model"""
        
        # Normalize sensor values
        normalized_value = (sensor_data.value - self._get_sensor_min(sensor_data.sensor_type)) / \
                          (self._get_sensor_max(sensor_data.sensor_type) - self._get_sensor_min(sensor_data.sensor_type))
        
        return [normalized_value]
    
    def _postprocess_prediction(self, prediction: Dict, sensor_data: SensorData) -> Dict:
        """Post-process AI prediction results"""
        
        return {
            "anomaly_detected": prediction.get("anomaly", False),
            "predicted_value": prediction.get("value", sensor_data.value),
            "trend": prediction.get("trend", "stable"),
            "recommendation": prediction.get("recommendation", "no_action")
        }
    
    def _get_sensor_min(self, sensor_type: str) -> float:
        """Get minimum value for sensor type"""
        sensor_ranges = {
            "temperature": -40.0,
            "humidity": 0.0,
            "pressure": 800.0,
            "light": 0.0
        }
        return sensor_ranges.get(sensor_type, 0.0)
    
    def _get_sensor_max(self, sensor_type: str) -> float:
        """Get maximum value for sensor type"""
        sensor_ranges = {
            "temperature": 100.0,
            "humidity": 100.0,
            "pressure": 1200.0,
            "light": 1000.0
        }
        return sensor_ranges.get(sensor_type, 100.0)
    
    async def _send_to_blockchain(self, sensor_data: SensorData):
        """Send sensor data to blockchain"""
        
        # This would integrate with blockchain manager
        pass
    
    async def _update_web3_dapp(self, sensor_data: SensorData):
        """Update Web3 dApp with sensor data"""
        
        # This would integrate with Web3 manager
        pass
    
    def _load_edge_models(self):
        """Load AI models for edge devices"""
        
        # Load TensorFlow Lite models for edge devices
        model_configs = {
            "device_001": {
                "model_path": "models/anomaly_detection.tflite",
                "input_shape": [1, 10],
                "output_shape": [1, 2]
            },
            "device_002": {
                "model_path": "models/prediction.tflite",
                "input_shape": [1, 5],
                "output_shape": [1, 1]
            }
        }
        
        for device_id, config in model_configs.items():
            try:
                # Load TensorFlow Lite model
                import tensorflow as tf
                interpreter = tf.lite.Interpreter(model_path=config["model_path"])
                interpreter.allocate_tensors()
                
                self.edge_ai_models[device_id] = {
                    "interpreter": interpreter,
                    "input_shape": config["input_shape"],
                    "output_shape": config["output_shape"],
                    "version": "1.0.0"
                }
                
                self.logger.info(f"Loaded edge AI model for device {device_id}")
                
            except Exception as e:
                self.logger.error(f"Failed to load model for device {device_id}: {str(e)}")
```

### 3. Blockchain Integration

```python
# blockchain_manager.py
import asyncio
import logging
from typing import Dict, List, Optional
from web3 import Web3
from eth_account import Account
import json

class BlockchainManager:
    def __init__(self, config: Dict):
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Initialize Web3 connection
        self.w3 = Web3(Web3.HTTPProvider(config["rpc_url"]))
        
        # Initialize account
        self.account = Account.from_key(config["private_key"])
        
        # Load smart contracts
        self.contracts = {}
        self._load_contracts()
    
    def _load_contracts(self):
        """Load smart contract ABIs and addresses"""
        
        # IoT Data Storage Contract
        with open("contracts/IoTDataStorage.json", "r") as f:
            iot_contract_abi = json.load(f)["abi"]
        
        self.contracts["iot_storage"] = {
            "abi": iot_contract_abi,
            "address": self.config["iot_storage_address"]
        }
        
        # AI Prediction Contract
        with open("contracts/AIPrediction.json", "r") as f:
            ai_contract_abi = json.load(f)["abi"]
        
        self.contracts["ai_prediction"] = {
            "abi": ai_contract_abi,
            "address": self.config["ai_prediction_address"]
        }
    
    async def store_data(self, device_id: str, sensor_data: Dict, ai_result: Dict) -> Dict:
        """Store sensor data and AI results on blockchain"""
        
        try:
            # Create contract instance
            contract = self.w3.eth.contract(
                address=self.contracts["iot_storage"]["address"],
                abi=self.contracts["iot_storage"]["abi"]
            )
            
            # Prepare data for blockchain
            data_hash = self.w3.keccak(
                text=json.dumps(sensor_data, sort_keys=True)
            ).hex()
            
            # Build transaction
            transaction = contract.functions.storeSensorData(
                device_id,
                data_hash,
                int(sensor_data["timestamp"]),
                int(sensor_data["value"] * 1000)  # Scale for integer storage
            ).build_transaction({
                'from': self.account.address,
                'gas': 200000,
                'gasPrice': self.w3.eth.gas_price,
                'nonce': self.w3.eth.get_transaction_count(self.account.address)
            })
            
            # Sign and send transaction
            signed_txn = self.w3.eth.account.sign_transaction(
                transaction, self.config["private_key"]
            )
            tx_hash = self.w3.eth.send_raw_transaction(signed_txn.rawTransaction)
            
            # Wait for transaction confirmation
            receipt = self.w3.eth.wait_for_transaction_receipt(tx_hash)
            
            return {
                "status": "success",
                "transaction_hash": tx_hash.hex(),
                "block_number": receipt.blockNumber,
                "gas_used": receipt.gasUsed
            }
            
        except Exception as e:
            self.logger.error(f"Blockchain storage failed: {str(e)}")
            return {
                "status": "failed",
                "error": str(e)
            }
    
    async def execute_ai_contract(self, device_id: str, ai_result: Dict) -> Dict:
        """Execute AI prediction smart contract"""
        
        try:
            # Create AI contract instance
            contract = self.w3.eth.contract(
                address=self.contracts["ai_prediction"]["address"],
                abi=self.contracts["ai_prediction"]["abi"]
            )
            
            # Execute smart contract with AI result
            transaction = contract.functions.processAIPrediction(
                device_id,
                ai_result.get("anomaly_detected", False),
                int(ai_result.get("confidence", 0) * 100),  # Scale to integer
                ai_result.get("recommendation", "no_action")
            ).build_transaction({
                'from': self.account.address,
                'gas': 150000,
                'gasPrice': self.w3.eth.gas_price,
                'nonce': self.w3.eth.get_transaction_count(self.account.address)
            })
            
            # Sign and send transaction
            signed_txn = self.w3.eth.account.sign_transaction(
                transaction, self.config["private_key"]
            )
            tx_hash = self.w3.eth.send_raw_transaction(signed_txn.rawTransaction)
            
            # Wait for confirmation
            receipt = self.w3.eth.wait_for_transaction_receipt(tx_hash)
            
            return {
                "status": "success",
                "transaction_hash": tx_hash.hex(),
                "block_number": receipt.blockNumber,
                "contract_result": await self._get_contract_result(contract, device_id)
            }
            
        except Exception as e:
            self.logger.error(f"AI contract execution failed: {str(e)}")
            return {
                "status": "failed",
                "error": str(e)
            }
    
    async def _get_contract_result(self, contract, device_id: str) -> Dict:
        """Get result from smart contract"""
        
        try:
            result = contract.functions.getDeviceStatus(device_id).call()
            return {
                "status": result[0],
                "last_update": result[1],
                "prediction_count": result[2]
            }
        except Exception as e:
            return {"error": str(e)}
```

### 4. Web3 Integration

```python
# web3_manager.py
import asyncio
import logging
from typing import Dict, List, Optional
from web3 import Web3
import ipfshttpclient
import json

class Web3Manager:
    def __init__(self, config: Dict):
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Initialize IPFS client
        self.ipfs_client = ipfshttpclient.connect(config["ipfs_url"])
        
        # Initialize Web3 for dApp
        self.w3 = Web3(Web3.HTTPProvider(config["web3_rpc_url"]))
        
        # Initialize dApp state
        self.dapp_state = {}
    
    async def update_dapp(self, device_id: str, ai_result: Dict, contract_result: Dict) -> Dict:
        """Update Web3 dApp with new data"""
        
        try:
            # Create dApp update
            update_data = {
                "device_id": device_id,
                "timestamp": asyncio.get_event_loop().time(),
                "ai_result": ai_result,
                "contract_result": contract_result,
                "blockchain_tx": contract_result.get("transaction_hash")
            }
            
            # Store on IPFS
            ipfs_hash = await self._store_on_ipfs(update_data)
            
            # Update dApp state
            await self._update_dapp_state(device_id, update_data, ipfs_hash)
            
            # Trigger dApp event
            await self._trigger_dapp_event(device_id, update_data)
            
            return {
                "status": "success",
                "ipfs_hash": ipfs_hash,
                "dapp_updated": True
            }
            
        except Exception as e:
            self.logger.error(f"Web3 dApp update failed: {str(e)}")
            return {
                "status": "failed",
                "error": str(e)
            }
    
    async def _store_on_ipfs(self, data: Dict) -> str:
        """Store data on IPFS"""
        
        try:
            # Convert data to JSON
            json_data = json.dumps(data, sort_keys=True)
            
            # Add to IPFS
            result = self.ipfs_client.add_json(json_data)
            
            return result
            
        except Exception as e:
            self.logger.error(f"IPFS storage failed: {str(e)}")
            raise
    
    async def _update_dapp_state(self, device_id: str, data: Dict, ipfs_hash: str):
        """Update dApp state"""
        
        if device_id not in self.dapp_state:
            self.dapp_state[device_id] = {
                "updates": [],
                "latest_ipfs_hash": None,
                "ai_predictions": [],
                "contract_interactions": []
            }
        
        # Add update to device state
        self.dapp_state[device_id]["updates"].append({
            "timestamp": data["timestamp"],
            "ipfs_hash": ipfs_hash,
            "ai_result": data["ai_result"],
            "contract_result": data["contract_result"]
        })
        
        # Keep only last 100 updates
        if len(self.dapp_state[device_id]["updates"]) > 100:
            self.dapp_state[device_id]["updates"] = self.dapp_state[device_id]["updates"][-100:]
        
        # Update latest IPFS hash
        self.dapp_state[device_id]["latest_ipfs_hash"] = ipfs_hash
        
        # Add AI prediction
        if data["ai_result"].get("status") == "success":
            self.dapp_state[device_id]["ai_predictions"].append({
                "timestamp": data["timestamp"],
                "prediction": data["ai_result"]["prediction"],
                "confidence": data["ai_result"]["confidence"]
            })
        
        # Add contract interaction
        if data["contract_result"].get("status") == "success":
            self.dapp_state[device_id]["contract_interactions"].append({
                "timestamp": data["timestamp"],
                "transaction_hash": data["contract_result"]["transaction_hash"],
                "block_number": data["contract_result"]["block_number"]
            })
    
    async def _trigger_dapp_event(self, device_id: str, data: Dict):
        """Trigger dApp event for real-time updates"""
        
        # This would trigger WebSocket events for dApp frontend
        event_data = {
            "type": "device_update",
            "device_id": device_id,
            "data": data
        }
        
        # Send to WebSocket server
        await self._send_websocket_event(event_data)
    
    async def _send_websocket_event(self, event_data: Dict):
        """Send event to WebSocket server"""
        
        # Implementation would depend on WebSocket server setup
        pass
    
    async def get_dapp_state(self, device_id: Optional[str] = None) -> Dict:
        """Get current dApp state"""
        
        if device_id:
            return self.dapp_state.get(device_id, {})
        else:
            return self.dapp_state
```

### 5. Smart Contract Examples

```solidity
// contracts/IoTDataStorage.sol
// SPDX-License-Identifier: MIT
pragma solidity ^0.8.0;

contract IoTDataStorage {
    struct SensorData {
        string deviceId;
        bytes32 dataHash;
        uint256 timestamp;
        uint256 value;
        bool exists;
    }
    
    mapping(string => SensorData[]) public deviceData;
    mapping(string => uint256) public deviceDataCount;
    
    event DataStored(string deviceId, bytes32 dataHash, uint256 timestamp);
    
    function storeSensorData(
        string memory deviceId,
        bytes32 dataHash,
        uint256 timestamp,
        uint256 value
    ) public {
        SensorData memory newData = SensorData({
            deviceId: deviceId,
            dataHash: dataHash,
            timestamp: timestamp,
            value: value,
            exists: true
        });
        
        deviceData[deviceId].push(newData);
        deviceDataCount[deviceId]++;
        
        emit DataStored(deviceId, dataHash, timestamp);
    }
    
    function getDeviceDataCount(string memory deviceId) public view returns (uint256) {
        return deviceDataCount[deviceId];
    }
    
    function getDeviceData(string memory deviceId, uint256 index) public view returns (
        string memory,
        bytes32,
        uint256,
        uint256,
        bool
    ) {
        require(index < deviceDataCount[deviceId], "Index out of bounds");
        SensorData memory data = deviceData[deviceId][index];
        return (data.deviceId, data.dataHash, data.timestamp, data.value, data.exists);
    }
}

// contracts/AIPrediction.sol
// SPDX-License-Identifier: MIT
pragma solidity ^0.8.0;

contract AIPrediction {
    struct Prediction {
        string deviceId;
        bool anomalyDetected;
        uint8 confidence;
        string recommendation;
        uint256 timestamp;
        bool exists;
    }
    
    mapping(string => Prediction[]) public devicePredictions;
    mapping(string => uint256) public devicePredictionCount;
    
    event PredictionProcessed(
        string deviceId,
        bool anomalyDetected,
        uint8 confidence,
        string recommendation
    );
    
    function processAIPrediction(
        string memory deviceId,
        bool anomalyDetected,
        uint8 confidence,
        string memory recommendation
    ) public {
        Prediction memory newPrediction = Prediction({
            deviceId: deviceId,
            anomalyDetected: anomalyDetected,
            confidence: confidence,
            recommendation: recommendation,
            timestamp: block.timestamp,
            exists: true
        });
        
        devicePredictions[deviceId].push(newPrediction);
        devicePredictionCount[deviceId]++;
        
        emit PredictionProcessed(deviceId, anomalyDetected, confidence, recommendation);
    }
    
    function getDeviceStatus(string memory deviceId) public view returns (
        bool,
        uint256,
        uint256
    ) {
        uint256 count = devicePredictionCount[deviceId];
        if (count == 0) {
            return (false, 0, 0);
        }
        
        Prediction memory latest = devicePredictions[deviceId][count - 1];
        return (latest.anomalyDetected, latest.timestamp, count);
    }
}
```

### 6. Frontend dApp Interface

```typescript
// components/IoTWeb3Dashboard.tsx
import React, { useState, useEffect } from 'react';
import { Card, Title, Text, Button, Badge } from '@tremor/react';
import { LineChart, BarChart } from 'recharts';
import { ethers } from 'ethers';

interface IoTDevice {
  id: string;
  name: string;
  status: string;
  last_update: string;
  sensor_data: Array<{
    timestamp: string;
    value: number;
    unit: string;
  }>;
  ai_predictions: Array<{
    timestamp: string;
    anomaly_detected: boolean;
    confidence: number;
    recommendation: string;
  }>;
  blockchain_data: {
    transaction_count: number;
    last_transaction: string;
    ipfs_hash: string;
  };
}

export const IoTWeb3Dashboard: React.FC = () => {
  const [devices, setDevices] = useState<IoTDevice[]>([]);
  const [selectedDevice, setSelectedDevice] = useState<string | null>(null);
  const [web3Provider, setWeb3Provider] = useState<any>(null);
  const [loading, setLoading] = useState(true);
  
  useEffect(() => {
    const initializeWeb3 = async () => {
      try {
        // Check if MetaMask is installed
        if (typeof window.ethereum !== 'undefined') {
          const provider = new ethers.providers.Web3Provider(window.ethereum);
          await provider.send("eth_requestAccounts", []);
          setWeb3Provider(provider);
        }
      } catch (error) {
        console.error('Failed to initialize Web3:', error);
      }
    };
    
    initializeWeb3();
    fetchDevices();
  }, []);
  
  const fetchDevices = async () => {
    try {
      const response = await fetch('/api/iot/devices');
      const data = await response.json();
      setDevices(data);
    } catch (error) {
      console.error('Error fetching devices:', error);
    } finally {
      setLoading(false);
    }
  };
  
  const connectWallet = async () => {
    if (web3Provider) {
      try {
        const signer = web3Provider.getSigner();
        const address = await signer.getAddress();
        console.log('Connected wallet:', address);
      } catch (error) {
        console.error('Failed to connect wallet:', error);
      }
    }
  };
  
  const viewOnBlockchain = async (deviceId: string) => {
    if (web3Provider) {
      try {
        const signer = web3Provider.getSigner();
        const contract = new ethers.Contract(
          process.env.REACT_APP_CONTRACT_ADDRESS!,
          ['function getDeviceDataCount(string) view returns (uint256)'],
          signer
        );
        
        const count = await contract.getDeviceDataCount(deviceId);
        console.log(`Device ${deviceId} has ${count} data points on blockchain`);
      } catch (error) {
        console.error('Failed to view on blockchain:', error);
      }
    }
  };
  
  if (loading) {
    return <div>Loading IoT Web3 Dashboard...</div>;
  }
  
  return (
    <div className="max-w-7xl mx-auto p-6">
      <Title className="text-3xl font-bold mb-6">
        IoT + AI + Blockchain Dashboard
      </Title>
      
      {/* Web3 Connection */}
      <Card className="mb-8">
        <Title>Web3 Connection</Title>
        <div className="flex gap-4 mt-4">
          <Button
            onClick={connectWallet}
            disabled={!web3Provider}
            color={web3Provider ? "green" : "gray"}
          >
            {web3Provider ? "Connect Wallet" : "MetaMask Required"}
          </Button>
          <Text>
            {web3Provider ? "Web3 Connected" : "Please install MetaMask"}
          </Text>
        </div>
      </Card>
      
      {/* Device Grid */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
        {devices.map(device => (
          <Card key={device.id} className="cursor-pointer hover:shadow-lg">
            <div className="flex justify-between items-start mb-4">
              <Title>{device.name}</Title>
              <Badge color={device.status === 'online' ? 'green' : 'red'}>
                {device.status}
              </Badge>
            </div>
            
            <Text className="text-sm text-gray-600 mb-4">
              Last Update: {new Date(device.last_update).toLocaleString()}
            </Text>
            
            {/* Sensor Data Chart */}
            <div className="mb-4">
              <Text className="text-sm font-medium mb-2">Sensor Data</Text>
              <LineChart
                data={device.sensor_data.slice(-10)}
                dataKey="timestamp"
                valueKey="value"
                height={100}
              />
            </div>
            
            {/* AI Predictions */}
            <div className="mb-4">
              <Text className="text-sm font-medium mb-2">AI Predictions</Text>
              {device.ai_predictions.slice(-3).map((pred, index) => (
                <div key={index} className="flex justify-between text-xs">
                  <span>{pred.anomaly_detected ? "⚠️ Anomaly" : "✅ Normal"}</span>
                  <span>{pred.confidence}% confidence</span>
                </div>
              ))}
            </div>
            
            {/* Blockchain Data */}
            <div className="mb-4">
              <Text className="text-sm font-medium mb-2">Blockchain</Text>
              <div className="text-xs space-y-1">
                <div>Transactions: {device.blockchain_data.transaction_count}</div>
                <div>IPFS Hash: {device.blockchain_data.ipfs_hash.slice(0, 10)}...</div>
              </div>
            </div>
            
            <Button
              size="xs"
              onClick={() => viewOnBlockchain(device.id)}
              disabled={!web3Provider}
            >
              View on Blockchain
            </Button>
          </Card>
        ))}
      </div>
      
      {/* Real-time Updates */}
      <Card className="mt-8">
        <Title>Real-time Updates</Title>
        <div className="mt-4">
          <Text>Live sensor data and AI predictions will appear here...</Text>
        </div>
      </Card>
    </div>
  );
};
```

---

## 🔧 Implementation Guide

### Phase 1: IoT Setup (Week 1-2)
1. **Hardware Setup**
   - Raspberry Pi configuration
   - Sensor integration
   - MQTT broker setup

2. **Edge AI Development**
   - TensorFlow Lite model training
   - Model optimization for edge devices
   - Real-time inference setup

3. **Data Pipeline**
   - Sensor data collection
   - Data preprocessing
   - Real-time streaming

### Phase 2: Blockchain Integration (Week 3-4)
1. **Smart Contract Development**
   - Solidity contract writing
   - Contract testing and deployment
   - Gas optimization

2. **Blockchain Integration**
   - Web3.js integration
   - Transaction management
   - Event handling

3. **IPFS Integration**
   - Decentralized storage setup
   - Data pinning and retrieval
   - Content addressing

### Phase 3: Web3 dApp Development (Week 5-6)
1. **Frontend Development**
   - React dApp interface
   - MetaMask integration
   - Real-time updates

2. **Backend Services**
   - API development
   - WebSocket integration
   - Data synchronization

3. **User Experience**
   - Intuitive interface design
   - Mobile responsiveness
   - Performance optimization

### Phase 4: Production Deployment (Week 7-8)
1. **Infrastructure Setup**
   - Kubernetes deployment
   - Load balancing
   - Monitoring setup

2. **Security & Testing**
   - Smart contract auditing
   - Security testing
   - Performance testing

3. **Documentation & Training**
   - Complete documentation
   - User guides
   - Deployment guides

---

## 📊 Evaluation Criteria

### Technical Integration (35%)
- **IoT Integration**: Seamless sensor data collection and processing
- **Blockchain Integration**: Efficient smart contract execution
- **Web3 Integration**: Functional dApp with real-time updates
- **AI Integration**: Effective edge AI and blockchain AI

### Innovation & Cutting-edge (30%)
- **Emerging Technologies**: Effective use of IoT, Blockchain, Web3
- **Technical Innovation**: Novel integration approaches
- **Scalability**: Distributed and scalable architecture
- **Future-ready**: Forward-looking technology choices

### Performance & Reliability (25%)
- **System Performance**: Fast and reliable operation
- **Data Integrity**: Secure and accurate data handling
- **Uptime**: High availability and fault tolerance
- **Scalability**: Handles increased load effectively

### User Experience (10%)
- **Interface Design**: Intuitive and responsive dApp
- **Real-time Updates**: Smooth real-time data flow
- **Accessibility**: User-friendly interaction
- **Documentation**: Clear user guidance

---

## 🎯 Success Metrics

### Integration Metrics
- **IoT Performance**: 99%+ sensor data collection success
- **Blockchain Efficiency**: < 30s transaction confirmation
- **Web3 Functionality**: 100% dApp feature availability
- **AI Accuracy**: 90%+ prediction accuracy

### Technical Metrics
- **System Uptime**: 99.9%+ availability
- **Data Latency**: < 5s end-to-end processing
- **Blockchain Gas**: Optimized gas usage
- **IPFS Reliability**: 95%+ data availability

### Innovation Metrics
- **Technology Stack**: 4+ emerging technologies integrated
- **Scalability**: 10x capacity increase capability
- **Future-readiness**: 80%+ future technology compatibility
- **Performance**: 50%+ improvement over baseline

### Business Metrics
- **User Adoption**: 70%+ user engagement
- **Data Volume**: 1000+ data points per day
- **Transaction Volume**: 100+ blockchain transactions per day
- **Cost Efficiency**: 40%+ cost reduction

---

## 🚀 Deployment Checklist

### Pre-Deployment
- [ ] IoT devices configured and tested
- [ ] Smart contracts audited and deployed
- [ ] Web3 dApp tested thoroughly
- [ ] Security audit completed
- [ ] Documentation updated

### Deployment
- [ ] IoT network deployed
- [ ] Blockchain contracts live
- [ ] Web3 dApp accessible
- [ ] Monitoring configured
- [ ] Health checks passing

### Post-Deployment
- [ ] Real-time monitoring active
- [ ] Performance optimization ongoing
- [ ] User feedback collected
- [ ] Continuous improvement plan
- [ ] Success metrics tracked

---

## 📚 Additional Resources

### Documentation
- [Ethereum Documentation](https://ethereum.org/developers/)
- [IPFS Documentation](https://docs.ipfs.io/)
- [TensorFlow Lite](https://www.tensorflow.org/lite)
- [Web3.js Documentation](https://web3js.org/)

### Tutorials
- [IoT with Blockchain](https://iot.org/)
- [Web3 Development](https://ethereum.org/developers/)
- [Edge AI Development](https://www.tensorflow.org/lite)

### Tools
- [Hardhat for Smart Contracts](https://hardhat.org/)
- [MetaMask for Web3](https://metamask.io/)
- [Grafana for Monitoring](https://grafana.com/)

This project demonstrates cutting-edge integration of AI/ML with emerging technologies, showcasing the future of decentralized, intelligent systems.

## 🏗️ Advanced Implementation Details

### 1. Federated Learning System

```python
# federated_learning.py
import asyncio
import logging
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
from datetime import datetime
import numpy as np
import torch
import torch.nn as nn
from cryptography.fernet import Fernet
import hashlib

@dataclass
class FederatedNode:
    """Federated learning node"""
    node_id: str
    device_type: str  # 'iot', 'edge', 'cloud'
    model_version: str
    data_size: int
    capabilities: List[str]
    location: Dict[str, float]

class FederatedLearningSystem:
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Initialize nodes
        self.nodes: Dict[str, FederatedNode] = {}
        self.global_model = None
        self.aggregation_history = []
        
        # Initialize encryption
        self.encryption_key = Fernet.generate_key()
        self.cipher_suite = Fernet(self.encryption_key)
        
        # Initialize blockchain for model versioning
        self.blockchain_manager = BlockchainManager(config['blockchain'])
    
    async def register_node(self, node: FederatedNode):
        """Register a federated learning node"""
        
        self.nodes[node.node_id] = node
        self.logger.info(f"Registered federated node: {node.node_id}")
        
        # Store node info on blockchain
        await self.blockchain_manager.store_node_info(node)
    
    async def start_federated_training(self, model_config: Dict[str, Any]):
        """Start federated learning training round"""
        
        self.logger.info("Starting federated learning training round")
        
        # Initialize global model
        self.global_model = self._create_model(model_config)
        
        # Distribute global model to nodes
        await self._distribute_model()
        
        # Start local training on nodes
        training_tasks = []
        for node_id in self.nodes.keys():
            task = asyncio.create_task(self._train_on_node(node_id))
            training_tasks.append(task)
        
        # Wait for all nodes to complete training
        results = await asyncio.gather(*training_tasks, return_exceptions=True)
        
        # Aggregate models
        aggregated_model = await self._aggregate_models(results)
        
        # Update global model
        self.global_model = aggregated_model
        
        # Store model version on blockchain
        model_hash = self._hash_model(aggregated_model)
        await self.blockchain_manager.store_model_version(model_hash, len(self.nodes))
        
        # Record aggregation
        self.aggregation_history.append({
            'timestamp': datetime.now(),
            'participating_nodes': len(self.nodes),
            'model_hash': model_hash,
            'performance_metrics': await self._evaluate_model(aggregated_model)
        })
        
        self.logger.info("Federated learning round completed")
    
    async def _distribute_model(self):
        """Distribute global model to all nodes"""
        
        for node_id, node in self.nodes.items():
            try:
                # Encrypt model for secure transmission
                model_bytes = self._model_to_bytes(self.global_model)
                encrypted_model = self.cipher_suite.encrypt(model_bytes)
                
                # Send to node
                await self._send_model_to_node(node_id, encrypted_model)
                
                self.logger.info(f"Distributed model to node: {node_id}")
                
            except Exception as e:
                self.logger.error(f"Failed to distribute model to {node_id}: {str(e)}")
    
    async def _train_on_node(self, node_id: str) -> Dict[str, Any]:
        """Train model on a specific node"""
        
        try:
            # Get local data for node
            local_data = await self._get_node_data(node_id)
            
            if len(local_data) == 0:
                return {
                    'node_id': node_id,
                    'status': 'no_data',
                    'model_update': None
                }
            
            # Create local model copy
            local_model = self._create_model_copy(self.global_model)
            
            # Train on local data
            trained_model = await self._train_local_model(local_model, local_data)
            
            # Calculate model update (difference from global)
            model_update = self._calculate_model_update(self.global_model, trained_model)
            
            # Encrypt model update
            encrypted_update = self.cipher_suite.encrypt(self._model_to_bytes(model_update))
            
            return {
                'node_id': node_id,
                'status': 'success',
                'model_update': encrypted_update,
                'data_size': len(local_data),
                'training_metrics': await self._get_training_metrics(trained_model, local_data)
            }
            
        except Exception as e:
            self.logger.error(f"Training failed on node {node_id}: {str(e)}")
            return {
                'node_id': node_id,
                'status': 'failed',
                'error': str(e)
            }
    
    async def _aggregate_models(self, results: List[Dict[str, Any]]) -> nn.Module:
        """Aggregate model updates from all nodes"""
        
        valid_updates = []
        total_data_size = 0
        
        for result in results:
            if result['status'] == 'success' and result['model_update'] is not None:
                # Decrypt model update
                decrypted_update = self.cipher_suite.decrypt(result['model_update'])
                model_update = self._bytes_to_model(decrypted_update)
                
                valid_updates.append({
                    'update': model_update,
                    'weight': result['data_size']
                })
                total_data_size += result['data_size']
        
        if not valid_updates:
            self.logger.warning("No valid model updates received")
            return self.global_model
        
        # Weighted average of model updates
        aggregated_model = self._create_model_copy(self.global_model)
        
        for param in aggregated_model.parameters():
            param.data.zero_()
        
        for update_info in valid_updates:
            weight = update_info['weight'] / total_data_size
            
            for param, update_param in zip(aggregated_model.parameters(), 
                                         update_info['update'].parameters()):
                param.data += update_param.data * weight
        
        return aggregated_model
    
    def _create_model(self, config: Dict[str, Any]) -> nn.Module:
        """Create neural network model"""
        
        model = nn.Sequential(
            nn.Linear(config['input_size'], 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, config['output_size'])
        )
        
        return model
    
    def _create_model_copy(self, model: nn.Module) -> nn.Module:
        """Create a copy of a model"""
        
        model_copy = type(model)()
        model_copy.load_state_dict(model.state_dict())
        return model_copy
    
    def _calculate_model_update(self, global_model: nn.Module, local_model: nn.Module) -> nn.Module:
        """Calculate the difference between global and local models"""
        
        update_model = self._create_model_copy(global_model)
        
        for global_param, local_param, update_param in zip(
            global_model.parameters(),
            local_model.parameters(),
            update_model.parameters()
        ):
            update_param.data = local_param.data - global_param.data
        
        return update_model
    
    async def _train_local_model(self, model: nn.Module, data: List[Dict]) -> nn.Module:
        """Train model on local data"""
        
        # Convert data to tensors
        X = torch.tensor([d['features'] for d in data], dtype=torch.float32)
        y = torch.tensor([d['label'] for d in data], dtype=torch.long)
        
        # Training setup
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        
        # Training loop
        model.train()
        for epoch in range(10):  # 10 epochs
            optimizer.zero_grad()
            outputs = model(X)
            loss = criterion(outputs, y)
            loss.backward()
            optimizer.step()
        
        return model
    
    async def _get_node_data(self, node_id: str) -> List[Dict]:
        """Get local data for a node"""
        
        # This would retrieve data from the specific node
        # For now, return mock data
        return [
            {
                'features': np.random.randn(10).tolist(),
                'label': np.random.randint(0, 3)
            }
            for _ in range(100)
        ]
    
    def _model_to_bytes(self, model: nn.Module) -> bytes:
        """Convert model to bytes"""
        
        import io
        buffer = io.BytesIO()
        torch.save(model.state_dict(), buffer)
        return buffer.getvalue()
    
    def _bytes_to_model(self, model_bytes: bytes) -> nn.Module:
        """Convert bytes back to model"""
        
        import io
        buffer = io.BytesIO(model_bytes)
        state_dict = torch.load(buffer)
        
        model = self._create_model({'input_size': 10, 'output_size': 3})
        model.load_state_dict(state_dict)
        return model
    
    def _hash_model(self, model: nn.Module) -> str:
        """Create hash of model for blockchain storage"""
        
        model_bytes = self._model_to_bytes(model)
        return hashlib.sha256(model_bytes).hexdigest()
    
    async def _evaluate_model(self, model: nn.Module) -> Dict[str, float]:
        """Evaluate model performance"""
        
        # This would evaluate on test data
        # For now, return mock metrics
        return {
            'accuracy': 0.85,
            'precision': 0.82,
            'recall': 0.88,
            'f1_score': 0.85
        }
    
    async def _get_training_metrics(self, model: nn.Module, data: List[Dict]) -> Dict[str, float]:
        """Get training metrics for a model"""
        
        # This would calculate actual training metrics
        # For now, return mock metrics
        return {
            'loss': 0.15,
            'accuracy': 0.87,
            'training_time': 2.5
        }
    
    async def _send_model_to_node(self, node_id: str, encrypted_model: bytes):
        """Send encrypted model to node"""
        
        # This would send to the actual node
        # For now, just log
        self.logger.info(f"Sent encrypted model to node {node_id}")
```

### 2. Advanced Smart Contract Integration

```python
# advanced_smart_contracts.py
import asyncio
import logging
from typing import Dict, List, Optional, Any
from web3 import Web3
from eth_account import Account
import json
import hashlib

class AdvancedSmartContractManager:
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Initialize Web3
        self.w3 = Web3(Web3.HTTPProvider(config["rpc_url"]))
        self.account = Account.from_key(config["private_key"])
        
        # Load advanced contracts
        self.contracts = {}
        self._load_advanced_contracts()
    
    def _load_advanced_contracts(self):
        """Load advanced smart contracts"""
        
        # Federated Learning Contract
        with open("contracts/FederatedLearning.json", "r") as f:
            fl_contract_abi = json.load(f)["abi"]
        
        self.contracts["federated_learning"] = {
            "abi": fl_contract_abi,
            "address": self.config["fl_contract_address"]
        }
        
        # AI Model Registry Contract
        with open("contracts/AIModelRegistry.json", "r") as f:
            model_registry_abi = json.load(f)["abi"]
        
        self.contracts["ai_model_registry"] = {
            "abi": model_registry_abi,
            "address": self.config["model_registry_address"]
        }
        
        # Data Marketplace Contract
        with open("contracts/DataMarketplace.json", "r") as f:
            marketplace_abi = json.load(f)["abi"]
        
        self.contracts["data_marketplace"] = {
            "abi": marketplace_abi,
            "address": self.config["marketplace_address"]
        }
    
    async def register_federated_node(self, node_info: Dict[str, Any]) -> Dict[str, Any]:
        """Register a federated learning node on blockchain"""
        
        try:
            contract = self.w3.eth.contract(
                address=self.contracts["federated_learning"]["address"],
                abi=self.contracts["federated_learning"]["abi"]
            )
            
            # Create node hash
            node_hash = self.w3.keccak(
                text=json.dumps(node_info, sort_keys=True)
            ).hex()
            
            # Build transaction
            transaction = contract.functions.registerNode(
                node_info["node_id"],
                node_hash,
                node_info["device_type"],
                node_info["data_size"],
                node_info["capabilities"]
            ).build_transaction({
                'from': self.account.address,
                'gas': 300000,
                'gasPrice': self.w3.eth.gas_price,
                'nonce': self.w3.eth.get_transaction_count(self.account.address)
            })
            
            # Sign and send
            signed_txn = self.w3.eth.account.sign_transaction(
                transaction, self.config["private_key"]
            )
            tx_hash = self.w3.eth.send_raw_transaction(signed_txn.rawTransaction)
            
            # Wait for confirmation
            receipt = self.w3.eth.wait_for_transaction_receipt(tx_hash)
            
            return {
                "status": "success",
                "transaction_hash": tx_hash.hex(),
                "node_hash": node_hash,
                "block_number": receipt.blockNumber
            }
            
        except Exception as e:
            self.logger.error(f"Failed to register federated node: {str(e)}")
            return {
                "status": "failed",
                "error": str(e)
            }
    
    async def store_model_version(self, model_hash: str, participating_nodes: int) -> Dict[str, Any]:
        """Store AI model version on blockchain"""
        
        try:
            contract = self.w3.eth.contract(
                address=self.contracts["ai_model_registry"]["address"],
                abi=self.contracts["ai_model_registry"]["abi"]
            )
            
            # Build transaction
            transaction = contract.functions.registerModelVersion(
                model_hash,
                participating_nodes,
                int(asyncio.get_event_loop().time())
            ).build_transaction({
                'from': self.account.address,
                'gas': 200000,
                'gasPrice': self.w3.eth.gas_price,
                'nonce': self.w3.eth.get_transaction_count(self.account.address)
            })
            
            # Sign and send
            signed_txn = self.w3.eth.account.sign_transaction(
                transaction, self.config["private_key"]
            )
            tx_hash = self.w3.eth.send_raw_transaction(signed_txn.rawTransaction)
            
            # Wait for confirmation
            receipt = self.w3.eth.wait_for_transaction_receipt(tx_hash)
            
            return {
                "status": "success",
                "transaction_hash": tx_hash.hex(),
                "model_hash": model_hash,
                "block_number": receipt.blockNumber
            }
            
        except Exception as e:
            self.logger.error(f"Failed to store model version: {str(e)}")
            return {
                "status": "failed",
                "error": str(e)
            }
    
    async def create_data_marketplace_listing(self, data_info: Dict[str, Any]) -> Dict[str, Any]:
        """Create a data marketplace listing"""
        
        try:
            contract = self.w3.eth.contract(
                address=self.contracts["data_marketplace"]["address"],
                abi=self.contracts["data_marketplace"]["abi"]
            )
            
            # Create data hash
            data_hash = self.w3.keccak(
                text=json.dumps(data_info, sort_keys=True)
            ).hex()
            
            # Build transaction
            transaction = contract.functions.createListing(
                data_info["owner"],
                data_hash,
                data_info["data_type"],
                data_info["price_wei"],
                data_info["description"]
            ).build_transaction({
                'from': self.account.address,
                'gas': 250000,
                'gasPrice': self.w3.eth.gas_price,
                'nonce': self.w3.eth.get_transaction_count(self.account.address)
            })
            
            # Sign and send
            signed_txn = self.w3.eth.account.sign_transaction(
                transaction, self.config["private_key"]
            )
            tx_hash = self.w3.eth.send_raw_transaction(signed_txn.rawTransaction)
            
            # Wait for confirmation
            receipt = self.w3.eth.wait_for_transaction_receipt(tx_hash)
            
            return {
                "status": "success",
                "transaction_hash": tx_hash.hex(),
                "data_hash": data_hash,
                "listing_id": await self._get_listing_id(contract, data_hash)
            }
            
        except Exception as e:
            self.logger.error(f"Failed to create data listing: {str(e)}")
            return {
                "status": "failed",
                "error": str(e)
            }
    
    async def _get_listing_id(self, contract, data_hash: str) -> int:
        """Get listing ID for data hash"""
        
        try:
            listing_id = contract.functions.getListingId(data_hash).call()
            return listing_id
        except Exception:
            return 0
    
    async def execute_ai_decision_contract(self, decision_data: Dict[str, Any]) -> Dict[str, Any]:
        """Execute AI-powered decision smart contract"""
        
        try:
            # This would execute a smart contract that makes decisions based on AI analysis
            contract = self.w3.eth.contract(
                address=self.config["ai_decision_address"],
                abi=self.config["ai_decision_abi"]
            )
            
            # Build transaction with AI decision
            transaction = contract.functions.executeDecision(
                decision_data["device_id"],
                decision_data["ai_prediction"],
                decision_data["confidence"],
                decision_data["action"]
            ).build_transaction({
                'from': self.account.address,
                'gas': 300000,
                'gasPrice': self.w3.eth.gas_price,
                'nonce': self.w3.eth.get_transaction_count(self.account.address)
            })
            
            # Sign and send
            signed_txn = self.w3.eth.account.sign_transaction(
                transaction, self.config["private_key"]
            )
            tx_hash = self.w3.eth.send_raw_transaction(signed_txn.rawTransaction)
            
            # Wait for confirmation
            receipt = self.w3.eth.wait_for_transaction_receipt(tx_hash)
            
            return {
                "status": "success",
                "transaction_hash": tx_hash.hex(),
                "decision_executed": True,
                "block_number": receipt.blockNumber
            }
            
        except Exception as e:
            self.logger.error(f"Failed to execute AI decision: {str(e)}")
            return {
                "status": "failed",
                "error": str(e)
            }
```

### 3. Advanced Web3 dApp Features

```typescript
// advanced_web3_features.ts
import { ethers } from 'ethers';
import { Web3Provider } from '@ethersproject/providers';
import { IPFS } from 'ipfs-http-client';

interface AdvancedWeb3Config {
  ethereumRpcUrl: string;
  ipfsGateway: string;
  contractAddresses: {
    federatedLearning: string;
    modelRegistry: string;
    dataMarketplace: string;
  };
}

class AdvancedWeb3Manager {
  private provider: Web3Provider;
  private ipfs: IPFS;
  private contracts: Map<string, ethers.Contract>;
  private config: AdvancedWeb3Config;

  constructor(config: AdvancedWeb3Config) {
    this.config = config;
    this.contracts = new Map();
    this.initializeProvider();
    this.initializeIPFS();
  }

  private async initializeProvider() {
    if (typeof window.ethereum !== 'undefined') {
      this.provider = new ethers.providers.Web3Provider(window.ethereum);
      await this.provider.send("eth_requestAccounts", []);
    } else {
      throw new Error('MetaMask not found');
    }
  }

  private async initializeIPFS() {
    this.ipfs = IPFS.create({ url: this.config.ipfsGateway });
  }

  async deployFederatedModel(modelData: any, metadata: any) {
    try {
      // Store model on IPFS
      const modelCid = await this.ipfs.add(JSON.stringify(modelData));
      
      // Store metadata on IPFS
      const metadataCid = await this.ipfs.add(JSON.stringify(metadata));
      
      // Deploy to blockchain
      const contract = this.getContract('federatedLearning');
      const signer = this.provider.getSigner();
      
      const transaction = await contract.connect(signer).deployModel(
        modelCid.toString(),
        metadataCid.toString(),
        metadata.participatingNodes,
        metadata.performanceMetrics
      );
      
      const receipt = await transaction.wait();
      
      return {
        status: 'success',
        modelCid: modelCid.toString(),
        metadataCid: metadataCid.toString(),
        transactionHash: receipt.transactionHash,
        blockNumber: receipt.blockNumber
      };
    } catch (error) {
      console.error('Failed to deploy federated model:', error);
      return { status: 'failed', error: error.message };
    }
  }

  async createDataMarketplaceListing(dataInfo: any) {
    try {
      // Store data on IPFS
      const dataCid = await this.ipfs.add(JSON.stringify(dataInfo.data));
      
      // Create marketplace listing
      const contract = this.getContract('dataMarketplace');
      const signer = this.provider.getSigner();
      
      const transaction = await contract.connect(signer).createListing(
        dataInfo.owner,
        dataCid.toString(),
        dataInfo.dataType,
        ethers.utils.parseEther(dataInfo.price.toString()),
        dataInfo.description
      );
      
      const receipt = await transaction.wait();
      
      return {
        status: 'success',
        dataCid: dataCid.toString(),
        transactionHash: receipt.transactionHash,
        listingId: await this.getListingId(dataCid.toString())
      };
    } catch (error) {
      console.error('Failed to create data listing:', error);
      return { status: 'failed', error: error.message };
    }
  }

  async executeAIDecision(decisionData: any) {
    try {
      const contract = this.getContract('aiDecision');
      const signer = this.provider.getSigner();
      
      const transaction = await contract.connect(signer).executeDecision(
        decisionData.deviceId,
        decisionData.prediction,
        decisionData.confidence,
        decisionData.action
      );
      
      const receipt = await transaction.wait();
      
      return {
        status: 'success',
        transactionHash: receipt.transactionHash,
        decisionExecuted: true
      };
    } catch (error) {
      console.error('Failed to execute AI decision:', error);
      return { status: 'failed', error: error.message };
    }
  }

  async getFederatedModelHistory() {
    try {
      const contract = this.getContract('modelRegistry');
      const modelCount = await contract.getModelCount();
      
      const models = [];
      for (let i = 0; i < modelCount; i++) {
        const model = await contract.getModel(i);
        models.push({
          modelHash: model.modelHash,
          timestamp: model.timestamp.toNumber(),
          participatingNodes: model.participatingNodes.toNumber(),
          performanceMetrics: model.performanceMetrics
        });
      }
      
      return { status: 'success', models };
    } catch (error) {
      console.error('Failed to get model history:', error);
      return { status: 'failed', error: error.message };
    }
  }

  async getDataMarketplaceListings() {
    try {
      const contract = this.getContract('dataMarketplace');
      const listingCount = await contract.getListingCount();
      
      const listings = [];
      for (let i = 0; i < listingCount; i++) {
        const listing = await contract.getListing(i);
        listings.push({
          id: listing.id.toNumber(),
          owner: listing.owner,
          dataCid: listing.dataCid,
          dataType: listing.dataType,
          price: ethers.utils.formatEther(listing.price),
          description: listing.description,
          active: listing.active
        });
      }
      
      return { status: 'success', listings };
    } catch (error) {
      console.error('Failed to get marketplace listings:', error);
      return { status: 'failed', error: error.message };
    }
  }

  private getContract(contractName: string): ethers.Contract {
    if (!this.contracts.has(contractName)) {
      const address = this.config.contractAddresses[contractName];
      const abi = this.getContractABI(contractName);
      this.contracts.set(contractName, new ethers.Contract(address, abi, this.provider));
    }
    return this.contracts.get(contractName);
  }

  private getContractABI(contractName: string): any[] {
    // This would load the actual ABI from contract artifacts
    // For now, return mock ABI
    return [];
  }

  private async getListingId(dataCid: string): Promise<number> {
    try {
      const contract = this.getContract('dataMarketplace');
      return await contract.getListingId(dataCid);
    } catch (error) {
      return 0;
    }
  }
}

// React component for advanced Web3 features
export const AdvancedWeb3Dashboard: React.FC = () => {
  const [web3Manager, setWeb3Manager] = useState<AdvancedWeb3Manager | null>(null);
  const [federatedModels, setFederatedModels] = useState<any[]>([]);
  const [marketplaceListings, setMarketplaceListings] = useState<any[]>([]);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    const initializeWeb3 = async () => {
      try {
        const config: AdvancedWeb3Config = {
          ethereumRpcUrl: process.env.REACT_APP_ETHEREUM_RPC_URL!,
          ipfsGateway: process.env.REACT_APP_IPFS_GATEWAY!,
          contractAddresses: {
            federatedLearning: process.env.REACT_APP_FL_CONTRACT_ADDRESS!,
            modelRegistry: process.env.REACT_APP_MODEL_REGISTRY_ADDRESS!,
            dataMarketplace: process.env.REACT_APP_MARKETPLACE_ADDRESS!
          }
        };

        const manager = new AdvancedWeb3Manager(config);
        setWeb3Manager(manager);

        // Load initial data
        await loadFederatedModels(manager);
        await loadMarketplaceListings(manager);

      } catch (error) {
        console.error('Failed to initialize Web3:', error);
      } finally {
        setLoading(false);
      }
    };

    initializeWeb3();
  }, []);

  const loadFederatedModels = async (manager: AdvancedWeb3Manager) => {
    const result = await manager.getFederatedModelHistory();
    if (result.status === 'success') {
      setFederatedModels(result.models);
    }
  };

  const loadMarketplaceListings = async (manager: AdvancedWeb3Manager) => {
    const result = await manager.getDataMarketplaceListings();
    if (result.status === 'success') {
      setMarketplaceListings(result.listings);
    }
  };

  const deployModel = async () => {
    if (!web3Manager) return;

    const modelData = {
      // Mock model data
      weights: [1, 2, 3, 4, 5],
      architecture: 'neural_network',
      version: '1.0.0'
    };

    const metadata = {
      participatingNodes: 5,
      performanceMetrics: {
        accuracy: 0.85,
        precision: 0.82,
        recall: 0.88
      }
    };

    const result = await web3Manager.deployFederatedModel(modelData, metadata);
    if (result.status === 'success') {
      await loadFederatedModels(web3Manager);
    }
  };

  if (loading) {
    return <div>Loading Advanced Web3 Dashboard...</div>;
  }

  return (
    <div className="max-w-7xl mx-auto p-6">
      <Title className="text-3xl font-bold mb-6">
        Advanced Web3 + AI + IoT Dashboard
      </Title>

      {/* Federated Learning Models */}
      <Card className="mb-8">
        <Title>Federated Learning Models</Title>
        <div className="flex justify-between items-center mb-4">
          <Text>{federatedModels.length} models deployed</Text>
          <Button onClick={deployModel}>Deploy New Model</Button>
        </div>
        
        <div className="space-y-4">
          {federatedModels.map((model, index) => (
            <div key={index} className="border p-4 rounded">
              <div className="flex justify-between">
                <Text>Model Hash: {model.modelHash.slice(0, 10)}...</Text>
                <Text>Nodes: {model.participatingNodes}</Text>
              </div>
              <Text className="text-sm text-gray-600">
                Deployed: {new Date(model.timestamp * 1000).toLocaleString()}
              </Text>
            </div>
          ))}
        </div>
      </Card>

      {/* Data Marketplace */}
      <Card className="mb-8">
        <Title>Data Marketplace</Title>
        <div className="space-y-4">
          {marketplaceListings.map((listing, index) => (
            <div key={index} className="border p-4 rounded">
              <div className="flex justify-between">
                <Text>Data Type: {listing.dataType}</Text>
                <Text>Price: {listing.price} ETH</Text>
              </div>
              <Text className="text-sm text-gray-600">
                {listing.description}
              </Text>
              <Text className="text-xs text-gray-500">
                Owner: {listing.owner.slice(0, 10)}...
              </Text>
            </div>
          ))}
        </div>
      </Card>
    </div>
  );
};
```

## 📊 Business Case Studies

### Case Study 1: Decentralized AI Network

**Company**: AI Consortium
**Challenge**: Build a decentralized AI network with privacy-preserving federated learning
**Solution**: AI + IoT + Blockchain + Web3 Integration

1. **Initial State**
   - 50+ organizations with AI models
   - Data silos preventing collaboration
   - Privacy concerns limiting sharing
   - Inconsistent model performance
   - High computational costs

2. **Implementation**
   - Federated learning across distributed nodes
   - Blockchain-based model versioning
   - IoT devices for edge AI processing
   - Web3 dApp for model marketplace
   - Privacy-preserving encryption

3. **Results**
   - 40% improvement in model accuracy
   - 70% reduction in computational costs
   - 100% privacy compliance
   - 24/7 decentralized operation
   - 10x increase in model collaboration

4. **Key Learnings**
   - Federated learning enables collaboration without data sharing
   - Blockchain provides transparent model versioning
   - Edge AI reduces latency and bandwidth
   - Web3 creates new business models
   - Privacy-first design builds trust

### Case Study 2: Smart City IoT Platform

**Company**: Urban Technology Solutions
**Challenge**: Build intelligent city infrastructure with decentralized control
**Solution**: IoT + AI + Blockchain Integration

1. **Initial State**
   - 1000+ IoT sensors across city
   - Centralized control system
   - High latency response times
   - Security vulnerabilities
   - Limited scalability

2. **Implementation**
   - Edge AI processing on IoT devices
   - Blockchain for secure data storage
   - Smart contracts for automated decisions
   - Web3 dApp for citizen interaction
   - Federated learning for traffic optimization

3. **Results**
   - 80% reduction in response time
   - 90% improvement in traffic flow
   - 100% security compliance
   - 50% reduction in energy consumption
   - 24/7 autonomous operation

4. **Key Learnings**
   - Edge AI enables real-time decision making
   - Blockchain ensures data integrity
   - Smart contracts automate complex decisions
   - Web3 engages citizens in governance
   - Federated learning optimizes city-wide systems

### Case Study 3: Healthcare Data Marketplace

**Company**: Medical Data Exchange
**Challenge**: Create secure marketplace for medical AI data
**Solution**: AI + Blockchain + Web3 + IoT Integration

1. **Initial State**
   - Fragmented medical data sources
   - Privacy concerns preventing sharing
   - Inconsistent data quality
   - Limited AI model training data
   - High regulatory compliance costs

2. **Implementation**
   - Blockchain-based data marketplace
   - Federated learning for model training
   - IoT devices for patient monitoring
   - Web3 dApp for data trading
   - Privacy-preserving AI algorithms

3. **Results**
   - 60% increase in available training data
   - 95% improvement in model accuracy
   - 100% regulatory compliance
   - 40% reduction in data acquisition costs
   - New revenue streams for data providers

4. **Key Learnings**
   - Blockchain enables secure data trading
   - Federated learning preserves privacy
   - IoT provides real-time health data
   - Web3 creates new business models
   - Privacy-first design is essential

## 📚 Portfolio Building Guide

### 1. Technical Documentation

Create comprehensive documentation covering:
- Multi-technology integration architecture
- Federated learning implementation
- Blockchain smart contract design
- Web3 dApp development
- IoT edge AI processing
- Privacy and security considerations

### 2. System Architecture Showcase

Highlight key architectural components:
- Federated learning network design
- Blockchain integration patterns
- Web3 dApp architecture
- IoT edge processing
- Smart contract automation
- Privacy-preserving mechanisms

### 3. Code Samples and Demonstrations

Showcase key implementations:
- Federated learning algorithms
- Smart contract development
- Web3 dApp features
- IoT edge AI processing
- Blockchain integration
- Privacy-preserving techniques

### 4. Case Study Presentations

Develop presentations covering:
- Business requirements and constraints
- Technical solution architecture
- Implementation challenges and solutions
- Results and impact analysis
- Lessons learned and best practices
- Future enhancement opportunities

### 5. GitHub Repository

Maintain a professional repository with:
- Clean, well-documented code structure
- Comprehensive README and documentation
- Performance benchmarks and metrics
- Deployment guides and examples
- Testing frameworks and examples
- Monitoring and observability tools

## 🎓 Assessment Criteria

### 1. Technical Integration (40%)

- [ ] Complete multi-technology integration
- [ ] Federated learning implementation
- [ ] Blockchain smart contract development
- [ ] Web3 dApp functionality
- [ ] IoT edge AI processing

### 2. Innovation & Cutting-edge (30%)

- [ ] Emerging technology integration
- [ ] Novel architectural patterns
- [ ] Privacy-preserving techniques
- [ ] Decentralized AI approaches
- [ ] Future-ready design

### 3. Performance & Reliability (20%)

- [ ] System performance optimization
- [ ] Data integrity and security
- [ ] Scalability and fault tolerance
- [ ] Privacy compliance
- [ ] Real-time processing capabilities

### 4. User Experience (10%)

- [ ] Intuitive Web3 interface
- [ ] Real-time data visualization
- [ ] Cross-platform compatibility
- [ ] Accessibility and usability
- [ ] Comprehensive documentation

## 🔬 Research Integration

### 1. Latest Research Papers

1. "Federated Learning for IoT Systems" (2024)
   - Privacy-preserving techniques
   - Edge AI optimization
   - Distributed training strategies

2. "Blockchain for AI Model Management" (2024)
   - Decentralized model versioning
   - Smart contract automation
   - Trust and transparency

3. "Web3 AI Applications" (2024)
   - Decentralized AI marketplaces
   - Tokenized AI services
   - Community-driven AI development

### 2. Future Trends

1. **Decentralized AI**
   - Federated learning networks
   - Privacy-preserving AI
   - Community-driven model development

2. **Tokenized AI Services**
   - AI model marketplaces
   - Data tokenization
   - Incentivized collaboration

3. **Edge AI Evolution**
   - Advanced edge processing
   - Federated edge learning
   - Autonomous edge decisions

## 🚀 Next Steps

1. **Advanced Features**
   - Advanced federated learning
   - Cross-chain AI integration
   - Advanced privacy techniques

2. **Platform Expansion**
   - Additional blockchain networks
   - New IoT device types
   - Advanced Web3 features

3. **Research Opportunities**
   - Novel federated learning approaches
   - Advanced blockchain AI integration
   - Privacy-preserving AI techniques

4. **Community Building**
   - Open source contributions
   - Documentation improvements
   - Tutorial development

## 📈 Success Metrics

### 1. Technical Metrics

- Federated learning accuracy > 90%
- Blockchain transaction success > 99%
- Web3 dApp uptime > 99.9%
- IoT edge processing latency < 100ms
- Privacy compliance > 100%

### 2. Innovation Metrics

- 4+ emerging technologies integrated
- Novel architectural patterns
- Advanced privacy techniques
- Future-ready design
- Research contributions

### 3. Performance Metrics

- 10x throughput increase
- 50% latency reduction
- 70% cost efficiency
- 100x scalability improvement
- 90%+ user satisfaction

### 4. Business Metrics

- 60% cost reduction
- 40% revenue increase
- 80% customer satisfaction
- 100x capacity increase
- Positive ROI in 6 months

## 🏆 Certification Requirements

1. **Implementation**
   - Complete multi-technology integration
   - Federated learning deployment
   - Blockchain smart contracts
   - Web3 dApp development
   - IoT edge AI processing

2. **Evaluation**
   - Technical assessment
   - Performance testing
   - Security audit
   - Privacy compliance review

3. **Presentation**
   - Architecture overview
   - Implementation details
   - Results analysis
   - Future roadmap

4. **Portfolio**
   - Project documentation
   - Code samples
   - Case studies
   - Performance benchmarks 