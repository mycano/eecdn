#include <iostream>
#include <fstream>
#include <cassert>
#include <string>

#include "ns3/command-line.h"
#include "ns3/config.h"
#include "ns3/core-module.h"
#include "ns3/internet-stack-helper.h"
#include "ns3/ipv4-address.h"
#include "ns3/ipv4-address-helper.h"
#include "ns3/ipv4-global-routing-helper.h"
#include "ns3/mobility-module.h"
#include "ns3/packet-sink-helper.h"
#include "ns3/packet-sink.h"
#include "ns3/on-off-helper.h"
#include "ns3/udp-client-server-helper.h"
#include "ns3/ssid.h"
#include "ns3/yans-wifi-channel.h"
#include "ns3/yans-wifi-helper.h"
#include "ns3/flow-monitor-module.h"

using namespace ns3;
using namespace std;

std::string wifiMode = "HeMcs0";

Ipv4Address getIPAddress (int deviceNo, int phase)
{
    std::string ipAddress;
    if (phase == 2)
    {
        ipAddress = "10.2.";
    }
    else
    {
        ipAddress = "10.1.";
    }
    ipAddress.append (std::to_string (deviceNo + 1));
    ipAddress.append (".0");
    return Ipv4Address (ipAddress.c_str ());
}

int main(int argc, char *argv[])
{
    // can change
    double simulationTime = 10;
    int channelWidth = 20;
    bool udp = true;
    double frequency = 5.0;
    int gi = 800;
    uint32_t payloadSize;
    double simulatorTxStartTime = 1.0;
    // cannot change
    int mobileNum = 4;
    int relayNum = 4;
    CommandLine cmd;
    cmd.AddValue ("mobilenum", "the number of mobile device", mobileNum);
    cmd.AddValue ("relaynum", "the number of relay device", relayNum);
    cmd.AddValue ("gi", "gi", gi);
    cmd.AddValue ("channelwidth", "channelwidth", channelWidth);
    cmd.Parse (argc, argv);
    long datasize[mobileNum];
    double txPower[mobileNum];
    double txRelayPower[mobileNum];
    double mobileTopology[mobileNum][2];
    double relayTopology[relayNum][2];
    double edgeTopology[1][2] = {{50, 50}};
    int policy[mobileNum];
    int match[mobileNum];
    std::ifstream in("/home/myyao/Desktop/ns-allinone-3.29/ns-3.29/scratch/EXPER/topo.txt");
    for (int i=0; i<mobileNum; i++)
        in >> datasize[i];
    for (int i=0; i<mobileNum; i++)
        in >> txPower[i];
    for (int i=0; i<mobileNum; i++)
        in >> txRelayPower[i];
    for (int i=0; i<mobileNum; i++)
    {
        in >> mobileTopology[i][0];
        in >> mobileTopology[i][1];
    }
    for (int j=0; j<relayNum; j++)
    {
        in >> relayTopology[j][0];
        in >> relayTopology[j][1];
    }
    for (int i=0; i<mobileNum; i++)
        in >> policy[i];
    for (int i=0; i<mobileNum; i++)
        in >> match[i];
    in.close();
    // std::cout << "mobile num: " << mobileNum << ", relay num: " << relayNum << std::endl;
    // cout << "datasize: ";
    // for (int i=0; i<mobileNum; i++)
    //     cout << datasize[i] << "\t";
    // cout << endl;
    // cout << "txpower: ";
    // for (int i=0; i<mobileNum; i++)
    //     cout << txPower[i] << "\t";
    // cout << endl;
    // cout << "txpower_relay: ";
    // for (int i=0; i<mobileNum; i++)
    //     cout << txRelayPower[i] << "\t";
    // cout << endl;
    // cout << "mobile topo: ";
    // for (int i=0; i<mobileNum; i++)
    //     cout << mobileTopology[i][0] << "\t" << mobileTopology[i][1] << "\t";
    // cout << endl;
    // cout << "relay topo: ";
    // for (int i=0; i<mobileNum; i++)
    //     cout << relayTopology[i][0] << "\t" << relayTopology[i][1] << "\t";
    // cout << endl;
    // cout << edgeTopology[0][0] << "\t" << edgeTopology[0][1] << endl;

    NodeContainer mobileNode;
    NodeContainer relayNode;
    NodeContainer edgeNode;
    mobileNode.Create (mobileNum);
    relayNode.Create (relayNum);
    edgeNode.Create (mobileNum);

    // configure the location
    MobilityHelper mobility;
    mobility.SetMobilityModel ("ns3::ConstantPositionMobilityModel");
    Ptr <ListPositionAllocator> positionAlloc;
    for (int i=0; i<mobileNum; i++)
    {
        positionAlloc = CreateObject <ListPositionAllocator> ();
        positionAlloc->Add (Vector (mobileTopology[i][0], mobileTopology[i][1], 0));
        mobility.SetPositionAllocator (positionAlloc);
        mobility.Install (mobileNode.Get (i));
    }
    for (int j=0; j<relayNum; j++)
    {
        positionAlloc = CreateObject <ListPositionAllocator> ();
        positionAlloc->Add (Vector (relayTopology[j][0], relayTopology[j][1], 0));
        mobility.SetPositionAllocator (positionAlloc);
        mobility.Install (relayNode.Get (j));
    }
    for (int i=0; i<mobileNum; i++)
    {
        positionAlloc = CreateObject <ListPositionAllocator> ();
        positionAlloc->Add (Vector (edgeTopology[0][0], edgeTopology[0][1], 0));
        mobility.SetPositionAllocator (positionAlloc);
        mobility.Install (edgeNode.Get (i));
    }
    // end of mobility

    // set trans bytes
    if (udp)
    {
        payloadSize = 1472; // bytes
    }
    else
    {
        payloadSize = 1448; // bytes
        Config::SetDefault("ns3::TcpSocket::SegmentSize", UintegerValue (payloadSize));
    }

    // configure the wireless
    NetDeviceContainer mobileNetDevice;
    NetDeviceContainer relayNetDevice;
    NetDeviceContainer edgeNetDevice;
    // use these two Ptr to represent the divce
    NetDeviceContainer sDevice; 
    NetDeviceContainer dDevice;
    // IP
    Ipv4AddressHelper address;
    Ipv4InterfaceContainer interface;
    // stack
    InternetStackHelper stack;
    stack.Install (mobileNode);
    stack.Install (relayNode);
    stack.Install (edgeNode);
    // store the IP address, which is used in the application
    Ipv4InterfaceContainer sourceAddr; 
    Ipv4InterfaceContainer endAddr; 
    // Ptr
    Ptr <NetDevice> sourceDevicePtr;
    Ptr <NetDevice> relayDevicePtr;
    // the index of matching relay
    int r;
    for (int i=0; i<mobileNum; i++)
    {
        sDevice = NetDeviceContainer ();
        dDevice = NetDeviceContainer ();
        YansWifiChannelHelper channel = YansWifiChannelHelper::Default ();
        YansWifiPhyHelper phy = YansWifiPhyHelper::Default ();
        phy.SetChannel (channel.Create ());
        // Set guard interval
        // phy.Set ("GuardInterval", TimeValue (NanoSeconds (gi)));

        WifiMacHelper mac;
        WifiHelper wifi;
        if (frequency == 5.0)
        {
            wifi.SetStandard (WIFI_PHY_STANDARD_80211ax_5GHZ);
        }
        else
        {
            wifi.SetStandard (WIFI_PHY_STANDARD_80211ax_2_4GHZ);
            Config::SetDefault ("ns3::LogDistancePropagationLossModel::ReferenceLoss", DoubleValue (40.046));
        }
        wifi.SetRemoteStationManager ("ns3::ConstantRateWifiManager","DataMode", StringValue (wifiMode),
                                            "ControlMode", StringValue (wifiMode));
        Ssid ssid = Ssid(std::to_string (i));

        mac.SetType ("ns3::StaWifiMac", "Ssid", SsidValue (ssid));
        mobileNetDevice.Add (wifi.Install (phy, mac, mobileNode.Get(i)));
        phy.Set ("TxPowerStart", DoubleValue (txPower[i]));
        phy.Set ("TxPowerEnd", DoubleValue (txPower[i]));
        if (policy[i] == 1)
        {
            sDevice.Add (wifi.Install (phy, mac, mobileNode.Get (i)));
            // ip address configure
            address.SetBase (getIPAddress (i, 1), "255.255.255.0");
            sourceDevicePtr = sDevice.Get (sDevice.GetN()-1);
            relayDevicePtr = sourceDevicePtr;
            interface = address.Assign (sourceDevicePtr);
            sourceAddr.Add (interface);
            endAddr.Add (interface);
        }
        else if (policy[i] == 2)
        {
            sDevice.Add (wifi.Install (phy, mac, mobileNode.Get (i)));
            mac.SetType ("ns3::ApWifiMac",
                           "EnableBeaconJitter", BooleanValue (false),
                           "Ssid", SsidValue (ssid));
            dDevice.Add (wifi.Install (phy, mac, edgeNode.Get (i)));
            // ip address configure
            address.SetBase (getIPAddress (i, 1), "255.255.255.0");
            sourceDevicePtr = sDevice.Get (sDevice.GetN()-1);
            relayDevicePtr = dDevice.Get (dDevice.GetN()-1);
            interface = address.Assign (sourceDevicePtr);
            sourceAddr.Add (interface);
            interface = address.Assign (relayDevicePtr);
            endAddr.Add (interface);
        }
        else if (policy[i] == 3)
        {

            sDevice.Add (wifi.Install (phy, mac, mobileNode.Get (i)));
            mac.SetType ("ns3::ApWifiMac",
                           "EnableBeaconJitter", BooleanValue (false),
                           "Ssid", SsidValue (ssid));
            r = match[i];
            dDevice.Add (wifi.Install (phy, mac, relayNode.Get (r)));
            // ip address configure
            address.SetBase (getIPAddress (i, 1), "255.255.255.0");
            sourceDevicePtr = sDevice.Get (sDevice.GetN()-1);
            relayDevicePtr = dDevice.Get (dDevice.GetN()-1);
            interface = address.Assign (sourceDevicePtr);
            sourceAddr.Add (interface);
            interface = address.Assign (relayDevicePtr);
            endAddr.Add (interface);
        }
        else
        {
            // phase 1
            sDevice.Add (wifi.Install (phy, mac, mobileNode.Get (i)));
            mac.SetType ("ns3::ApWifiMac",
                           "EnableBeaconJitter", BooleanValue (false),
                           "Ssid", SsidValue (ssid));
            r = match[i];
            dDevice.Add (wifi.Install (phy, mac, relayNode.Get (r)));
            // ip address configure
            address.SetBase (getIPAddress (i, 1), "255.255.255.0");
            sourceDevicePtr = sDevice.Get (sDevice.GetN()-1);
            relayDevicePtr = dDevice.Get (dDevice.GetN()-1);
            interface = address.Assign (sourceDevicePtr);
            sourceAddr.Add (interface);
            interface = address.Assign (relayDevicePtr);
            // phase 2
            ssid = Ssid (std::to_string (i+mobileNum*4+10));
            mac.SetType ("ns3::StaWifiMac", "Ssid", SsidValue (ssid));
            phy.SetChannel (channel.Create ());
            phy.Set ("TxPowerStart", DoubleValue (txRelayPower[i]));
            phy.Set ("TxPowerEnd", DoubleValue (txRelayPower[i]));
            sDevice.Add (wifi.Install (phy, mac, relayNode.Get (r)));
            mac.SetType ("ns3::ApWifiMac",
                           "EnableBeaconJitter", BooleanValue (false),
                           "Ssid", SsidValue (ssid));
            dDevice.Add (wifi.Install (phy, mac, edgeNode.Get(i)));
            // ip address configure
            address.SetBase (getIPAddress (i, 2), "255.255.255.0");
            sourceDevicePtr = sDevice.Get (sDevice.GetN()-1);
            relayDevicePtr = dDevice.Get (dDevice.GetN()-1);
            interface = address.Assign (sourceDevicePtr);
            interface = address.Assign (relayDevicePtr);
            endAddr.Add (interface);
        }
        
        // Set channel width
        Config::Set ("/NodeList/*/DeviceList/*/$ns3::WifiNetDevice/Phy/ChannelWidth", UintegerValue (channelWidth));
    }   

    /* Setting applications */
    uint16_t port = 1000;
    ApplicationContainer serverApp;
    if (udp)
    {
        port = 9;
        for (int i=0; i<mobileNum; i++)
        {
            if (policy[i] == 1)
            {
                continue;
            }
            UdpServerHelper server (port);
            if (policy[i] == 2)
            {
                serverApp = server.Install (edgeNode.Get (i));
            }
            else if (policy[i] == 3)
            {
                r = match[i];
                serverApp = server.Install (relayNode.Get (r));
            }
            else
            {
                serverApp = server.Install (edgeNode.Get (i));
            }
            serverApp.Start (Seconds (0.0));
            serverApp.Stop (Seconds (simulationTime));

            UdpClientHelper client (endAddr.GetAddress (i), port);
            client.SetAttribute ("MaxPackets", UintegerValue(int(datasize[i]/payloadSize)));
            client.SetAttribute ("Interval", TimeValue (Time ("0.000001"))); //packets/s
            client.SetAttribute ("PacketSize", UintegerValue (payloadSize));
            ApplicationContainer clientApp = client.Install (mobileNode.Get (i));
            clientApp.Start (Seconds (simulatorTxStartTime));
            clientApp.Stop (Seconds (simulationTime));
        }
    }
    else
    {
        // TCP flow
        port = 50000;
        for (int i=0; i<mobileNum; i++)
        {
            if (policy[i] == 1)
                continue;
            Address localAddress (InetSocketAddress (Ipv4Address::GetAny (), port));
            PacketSinkHelper packetSinkHelper ("ns3::TcpSocketFactory", localAddress);
            if (policy[i] == 2)
            {
                serverApp = packetSinkHelper.Install (edgeNode.Get (i));
            }
            else if (policy[i] == 3)
            {
                r = match[i];
                serverApp = packetSinkHelper.Install (relayNode.Get (r));
            }
            else
            {
                serverApp = packetSinkHelper.Install (edgeNode.Get (i));
            }
            serverApp.Start (Seconds (0.0));
            serverApp.Stop (Seconds (simulationTime));

            OnOffHelper onoff ("ns3::TcpSocketFactory", Ipv4Address::GetAny ());
            onoff.SetAttribute ("OnTime",  StringValue ("ns3::ConstantRandomVariable[Constant=1]"));
            onoff.SetAttribute ("OffTime", StringValue ("ns3::ConstantRandomVariable[Constant=0]"));
            onoff.SetAttribute ("PacketSize", UintegerValue (payloadSize));
            onoff.SetAttribute ("DataRate", DataRateValue (1000000000)); //bit/s
            AddressValue remoteAddress (InetSocketAddress (endAddr.GetAddress (i), port));
            onoff.SetAttribute ("Remote", remoteAddress);
            onoff.SetAttribute ("MaxBytes", UintegerValue (datasize[i]));
            ApplicationContainer clientApp = onoff.Install (mobileNode.Get (i));
            clientApp.Start (Seconds (simulatorTxStartTime));
            clientApp.Stop (Seconds (simulationTime + 1));
        }
    }
    Ipv4GlobalRoutingHelper::PopulateRoutingTables ();
    
    double txTime[mobileNum];
    for (int i=0; i<mobileNum; i++)
    {
        txTime[i] = 0;
    }
    // flow detector
    FlowMonitorHelper flowmon;
    Ptr <FlowMonitor> monitor = flowmon.InstallAll ();
    Simulator::Stop (Seconds (simulationTime+0.1));
    Simulator::Run ();

    Ptr<Ipv4FlowClassifier> classifier = DynamicCast<Ipv4FlowClassifier> (flowmon.GetClassifier ());
    FlowMonitor::FlowStatsContainer stats = monitor->GetFlowStats ();
    for (std::map<FlowId, FlowMonitor::FlowStats>::const_iterator i = stats.begin (); i!=stats.end (); i++)
    {
        Ipv4FlowClassifier::FiveTuple t = classifier->FindFlow (i->first);
        for (unsigned int k=0; k<sourceAddr.GetN (); k++)
        {
            if (t.destinationAddress == endAddr.GetAddress (k))
            {
                txTime[k] = (i->second.timeLastRxPacket.GetDouble ()) / pow (10, 9) - simulatorTxStartTime;
            }
        }
    }
    Simulator::Destroy ();

    // cout << "txTime: ";
    // for (int i=0; i<mobileNum; i++)
    // {
    //     cout << txTime[i] << "\t";
    // }
    // cout << endl;
    ofstream outTxt;
    outTxt.open("/home/myyao/Desktop/ns-allinone-3.29/ns-3.29/scratch/EXPER/time.txt", ios::out);
    for (int i=0; i<mobileNum; i++)
    {
        outTxt << txTime[i];
        outTxt << " ";
    }
    outTxt.close();
    return 0;
}
