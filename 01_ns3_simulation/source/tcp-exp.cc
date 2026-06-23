/*
 * SPDX-License-Identifier: GPL-2.0-only
 *
 * TCP throughput experiment with configurable point-to-point links.
 *
 * Topology:
 *
 *   n0 -------- n1 -------- n2 -------- n3
 * sender      router      router      receiver
 *
 * Each link has independently configurable bandwidth, one-way delay, and
 * packet loss rate. A long-lived TCP BulkSendApplication runs from n0 to n3,
 * and the receiver-side PacketSink byte count is used to compute throughput.
 */

#include "ns3/applications-module.h"
#include "ns3/core-module.h"
#include "ns3/flow-monitor-module.h"
#include "ns3/internet-module.h"
#include "ns3/network-module.h"
#include "ns3/point-to-point-module.h"

#include <algorithm>
#include <iomanip>
#include <iostream>
#include <string>

using namespace ns3;

NS_LOG_COMPONENT_DEFINE("TcpExp");

namespace
{

class RandomIntervalTcpApplication : public Application
{
  public:
    RandomIntervalTcpApplication();
    ~RandomIntervalTcpApplication() override;

    void Setup(Address peer,
               uint32_t packetSize,
               uint64_t maxBytes,
               Time minInterval,
               Time maxInterval);

  private:
    void StartApplication() override;
    void StopApplication() override;
    void SendPacket();
    void ScheduleNextTx();

    Ptr<Socket> m_socket;
    Address m_peer;
    EventId m_sendEvent;
    Ptr<UniformRandomVariable> m_intervalRng;
    uint32_t m_packetSize;
    uint64_t m_maxBytes;
    uint64_t m_totalTx;
    Time m_minInterval;
    Time m_maxInterval;
    bool m_running;
};

RandomIntervalTcpApplication::RandomIntervalTcpApplication()
    : m_socket(nullptr),
      m_intervalRng(CreateObject<UniformRandomVariable>()),
      m_packetSize(1024),
      m_maxBytes(0),
      m_totalTx(0),
      m_minInterval(MilliSeconds(1)),
      m_maxInterval(MilliSeconds(10)),
      m_running(false)
{
}

RandomIntervalTcpApplication::~RandomIntervalTcpApplication()
{
    m_socket = nullptr;
}

void
RandomIntervalTcpApplication::Setup(Address peer,
                                    uint32_t packetSize,
                                    uint64_t maxBytes,
                                    Time minInterval,
                                    Time maxInterval)
{
    m_peer = peer;
    m_packetSize = packetSize;
    m_maxBytes = maxBytes;
    m_minInterval = minInterval;
    m_maxInterval = maxInterval;
    m_intervalRng->SetAttribute("Min", DoubleValue(m_minInterval.GetSeconds()));
    m_intervalRng->SetAttribute("Max", DoubleValue(m_maxInterval.GetSeconds()));
}

void
RandomIntervalTcpApplication::StartApplication()
{
    m_running = true;
    m_totalTx = 0;
    m_socket = Socket::CreateSocket(GetNode(), TcpSocketFactory::GetTypeId());
    m_socket->Connect(m_peer);
    SendPacket();
}

void
RandomIntervalTcpApplication::StopApplication()
{
    m_running = false;
    if (m_sendEvent.IsPending())
    {
        Simulator::Cancel(m_sendEvent);
    }
    if (m_socket)
    {
        m_socket->Close();
    }
}

void
RandomIntervalTcpApplication::SendPacket()
{
    if (!m_running)
    {
        return;
    }

    uint32_t bytesToSend = m_packetSize;
    if (m_maxBytes > 0)
    {
        if (m_totalTx >= m_maxBytes)
        {
            return;
        }
        bytesToSend = std::min<uint64_t>(m_packetSize, m_maxBytes - m_totalTx);
    }

    Ptr<Packet> packet = Create<Packet>(bytesToSend);
    int actual = m_socket->Send(packet);
    if (actual > 0)
    {
        m_totalTx += static_cast<uint64_t>(actual);
    }

    ScheduleNextTx();
}

void
RandomIntervalTcpApplication::ScheduleNextTx()
{
    if (!m_running)
    {
        return;
    }
    if (m_maxBytes > 0 && m_totalTx >= m_maxBytes)
    {
        return;
    }

    double nextInterval = m_intervalRng->GetValue();
    m_sendEvent = Simulator::Schedule(Seconds(nextInterval),
                                      &RandomIntervalTcpApplication::SendPacket,
                                      this);
}

NetDeviceContainer
InstallPointToPointLink(NodeContainer nodes,
                        const std::string& dataRate,
                        const std::string& delay,
                        double lossRate)
{
    PointToPointHelper p2p;
    p2p.SetDeviceAttribute("DataRate", StringValue(dataRate));
    p2p.SetChannelAttribute("Delay", StringValue(delay));

    NetDeviceContainer devices = p2p.Install(nodes);

    if (lossRate > 0)
    {
        for (uint32_t i = 0; i < devices.GetN(); ++i)
        {
            Ptr<RateErrorModel> errorModel = CreateObject<RateErrorModel>();
            errorModel->SetAttribute("ErrorUnit", StringValue("ERROR_UNIT_PACKET"));
            errorModel->SetAttribute("ErrorRate", DoubleValue(lossRate));
            devices.Get(i)->SetAttribute("ReceiveErrorModel", PointerValue(errorModel));
        }
    }

    return devices;
}

void
PrintLinkConfig(const std::string& name,
                const std::string& dataRate,
                const std::string& delay,
                double lossRate)
{
    std::cout << name << ": rate=" << dataRate << ", delay=" << delay
              << ", lossRate=" << lossRate << std::endl;
}

} // namespace

int
main(int argc, char* argv[])
{
    std::string accessRate = "100Mbps";
    std::string accessDelay = "2ms";
    double accessLoss = 0.0;

    std::string bottleneckRate = "10Mbps";
    std::string bottleneckDelay = "20ms";
    double bottleneckLoss = 0.0;

    std::string egressRate = "100Mbps";
    std::string egressDelay = "2ms";
    double egressLoss = 0.0;

    std::string tcpType = "ns3::TcpCubic";
    uint32_t segmentSize = 1448;
    uint32_t tcpBufferSize = 16777216;
    std::string trafficMode = "bulk";
    double minSendIntervalMs = 1.0;
    double maxSendIntervalMs = 10.0;
    uint64_t maxApplicationBytes = 0;
    double startTime = 1.0;
    double stopTime = 20.0;
    bool enablePcap = false;

    CommandLine cmd(__FILE__);
    cmd.AddValue("accessRate", "Bandwidth of n0-n1 link", accessRate);
    cmd.AddValue("accessDelay", "One-way delay of n0-n1 link", accessDelay);
    cmd.AddValue("accessLoss", "Packet loss rate of n0-n1 link, e.g., 0.001", accessLoss);
    cmd.AddValue("bottleneckRate", "Bandwidth of n1-n2 link", bottleneckRate);
    cmd.AddValue("bottleneckDelay", "One-way delay of n1-n2 link", bottleneckDelay);
    cmd.AddValue("bottleneckLoss", "Packet loss rate of n1-n2 link, e.g., 0.001", bottleneckLoss);
    cmd.AddValue("egressRate", "Bandwidth of n2-n3 link", egressRate);
    cmd.AddValue("egressDelay", "One-way delay of n2-n3 link", egressDelay);
    cmd.AddValue("egressLoss", "Packet loss rate of n2-n3 link, e.g., 0.001", egressLoss);
    cmd.AddValue("tcpType", "TCP variant TypeId", tcpType);
    cmd.AddValue("segmentSize", "TCP segment size in bytes", segmentSize);
    cmd.AddValue("tcpBufferSize", "TCP send and receive buffer size in bytes", tcpBufferSize);
    cmd.AddValue("trafficMode", "Application traffic mode: bulk or random-interval", trafficMode);
    cmd.AddValue("minSendIntervalMs", "Minimum random packet send interval in ms", minSendIntervalMs);
    cmd.AddValue("maxSendIntervalMs", "Maximum random packet send interval in ms", maxSendIntervalMs);
    cmd.AddValue("maxApplicationBytes", "Maximum application bytes to send, 0 means unlimited", maxApplicationBytes);
    cmd.AddValue("startTime", "TCP application start time in seconds", startTime);
    cmd.AddValue("stopTime", "TCP application stop time in seconds", stopTime);
    cmd.AddValue("enablePcap", "Enable pcap tracing", enablePcap);
    cmd.Parse(argc, argv);

    if (stopTime <= startTime)
    {
        NS_FATAL_ERROR("stopTime must be greater than startTime");
    }
    if (trafficMode != "bulk" && trafficMode != "random-interval")
    {
        NS_FATAL_ERROR("trafficMode must be bulk or random-interval");
    }
    if (minSendIntervalMs <= 0 || maxSendIntervalMs <= 0 ||
        maxSendIntervalMs < minSendIntervalMs)
    {
        NS_FATAL_ERROR("Send interval bounds must be positive and max >= min");
    }

    Config::SetDefault("ns3::TcpL4Protocol::SocketType",
                       TypeIdValue(TypeId::LookupByName(tcpType)));
    Config::SetDefault("ns3::TcpSocket::SegmentSize", UintegerValue(segmentSize));
    Config::SetDefault("ns3::TcpSocket::SndBufSize", UintegerValue(tcpBufferSize));
    Config::SetDefault("ns3::TcpSocket::RcvBufSize", UintegerValue(tcpBufferSize));

    NodeContainer nodes;
    nodes.Create(4);

    NodeContainer n0n1(nodes.Get(0), nodes.Get(1));
    NodeContainer n1n2(nodes.Get(1), nodes.Get(2));
    NodeContainer n2n3(nodes.Get(2), nodes.Get(3));

    InternetStackHelper internet;
    internet.Install(nodes);

    NetDeviceContainer d0d1 =
        InstallPointToPointLink(n0n1, accessRate, accessDelay, accessLoss);
    NetDeviceContainer d1d2 =
        InstallPointToPointLink(n1n2, bottleneckRate, bottleneckDelay, bottleneckLoss);
    NetDeviceContainer d2d3 =
        InstallPointToPointLink(n2n3, egressRate, egressDelay, egressLoss);

    Ipv4AddressHelper ipv4;
    ipv4.SetBase("10.1.1.0", "255.255.255.0");
    ipv4.Assign(d0d1);

    ipv4.SetBase("10.1.2.0", "255.255.255.0");
    ipv4.Assign(d1d2);

    ipv4.SetBase("10.1.3.0", "255.255.255.0");
    Ipv4InterfaceContainer i2i3 = ipv4.Assign(d2d3);

    Ipv4GlobalRoutingHelper::PopulateRoutingTables();

    uint16_t port = 5000;
    PacketSinkHelper sinkHelper("ns3::TcpSocketFactory",
                                InetSocketAddress(Ipv4Address::GetAny(), port));
    ApplicationContainer sinkApp = sinkHelper.Install(nodes.Get(3));
    sinkApp.Start(Seconds(0.0));
    sinkApp.Stop(Seconds(stopTime + 1.0));

    ApplicationContainer senderApp;
    if (trafficMode == "bulk")
    {
        BulkSendHelper bulkSender("ns3::TcpSocketFactory",
                                  InetSocketAddress(i2i3.GetAddress(1), port));
        bulkSender.SetAttribute("MaxBytes", UintegerValue(maxApplicationBytes));
        senderApp = bulkSender.Install(nodes.Get(0));
    }
    else
    {
        Ptr<RandomIntervalTcpApplication> randomSender =
            CreateObject<RandomIntervalTcpApplication>();
        randomSender->Setup(InetSocketAddress(i2i3.GetAddress(1), port),
                            segmentSize,
                            maxApplicationBytes,
                            MilliSeconds(minSendIntervalMs),
                            MilliSeconds(maxSendIntervalMs));
        nodes.Get(0)->AddApplication(randomSender);
        senderApp.Add(randomSender);
    }
    senderApp.Start(Seconds(startTime));
    senderApp.Stop(Seconds(stopTime));

    FlowMonitorHelper flowmonHelper;
    Ptr<FlowMonitor> monitor = flowmonHelper.InstallAll();

    if (enablePcap)
    {
        PointToPointHelper p2p;
        p2p.EnablePcap("tcp-exp-access", d0d1);
        p2p.EnablePcap("tcp-exp-bottleneck", d1d2);
        p2p.EnablePcap("tcp-exp-egress", d2d3);
    }

    Simulator::Stop(Seconds(stopTime + 1.0));
    Simulator::Run();

    Ptr<PacketSink> sink = DynamicCast<PacketSink>(sinkApp.Get(0));
    uint64_t totalRxBytes = sink->GetTotalRx();
    double duration = stopTime - startTime;
    double throughputMbps = (totalRxBytes * 8.0) / duration / 1000000.0;

    monitor->CheckForLostPackets();
    FlowMonitor::FlowStatsContainer stats = monitor->GetFlowStats();

    std::cout << std::fixed << std::setprecision(6);
    std::cout << "TCP throughput experiment" << std::endl;
    PrintLinkConfig("n0-n1", accessRate, accessDelay, accessLoss);
    PrintLinkConfig("n1-n2", bottleneckRate, bottleneckDelay, bottleneckLoss);
    PrintLinkConfig("n2-n3", egressRate, egressDelay, egressLoss);
    std::cout << "tcpType=" << tcpType << ", segmentSize=" << segmentSize << " bytes"
              << std::endl;
    std::cout << "tcpBufferSize=" << tcpBufferSize << " bytes" << std::endl;
    std::cout << "trafficMode=" << trafficMode
              << ", minSendIntervalMs=" << minSendIntervalMs
              << ", maxSendIntervalMs=" << maxSendIntervalMs
              << ", maxApplicationBytes=" << maxApplicationBytes << std::endl;
    std::cout << "rxBytes=" << totalRxBytes << std::endl;
    std::cout << "averageThroughput=" << throughputMbps << " Mbit/s" << std::endl;

    for (const auto& flow : stats)
    {
        const FlowMonitor::FlowStats& stat = flow.second;
        if (stat.rxBytes == 0)
        {
            continue;
        }
        double flowDuration =
            (stat.timeLastRxPacket - stat.timeFirstTxPacket).GetSeconds();
        double flowThroughputMbps = 0.0;
        if (flowDuration > 0)
        {
            flowThroughputMbps = stat.rxBytes * 8.0 / flowDuration / 1000000.0;
        }
        std::cout << "flowId=" << flow.first << ", txPackets=" << stat.txPackets
                  << ", rxPackets=" << stat.rxPackets
                  << ", lostPackets=" << stat.lostPackets
                  << ", flowThroughput=" << flowThroughputMbps << " Mbit/s"
                  << std::endl;
    }

    Simulator::Destroy();
    return 0;
}
