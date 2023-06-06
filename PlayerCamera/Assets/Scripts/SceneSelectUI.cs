using System;
using System.Collections;
using System.Collections.Generic;
using System.Globalization;
using System.Linq;
using UnityEngine;
using UnityEngine.SceneManagement;
using UnityEngine.UI;
using Gyroscope = UnityEngine.InputSystem.Gyroscope;

#if URS_USE_AR_FOUNDATION
using UnityEngine.XR.ARFoundation;
#endif

namespace Unity.RenderStreaming.Samples
{
    internal enum SignalingType
    {
        WebSocket,
        Http,
        Furioos
    }

    internal class RenderStreamingSettings
    {
        public const int DefaultStreamWidth = 3840;
        public const int DefaultStreamHeight = 1920;

        private bool useDefaultSettings = false;
        private SignalingType signalingType = SignalingType.WebSocket;
        private string signalingAddress = "localhost";
        private int signalingInterval = 5000;
        private bool signalingSecured = false;
        private Vector2Int streamSize = new Vector2Int(DefaultStreamWidth, DefaultStreamHeight);
        private VideoCodecInfo receiverVideoCodec = null;
        private VideoCodecInfo senderVideoCodec = null;

        public bool UseDefaultSettings
        {
            get { return useDefaultSettings; }
            set { useDefaultSettings = value; }
        }

        public SignalingType SignalingType
        {
            get { return signalingType; }
            set { signalingType = value; }
        }

        public string SignalingAddress
        {
            get { return signalingAddress; }
            set { signalingAddress = value; }
        }

        public bool SignalingSecured
        {
            get { return signalingSecured; }
            set { signalingSecured = value; }
        }

        public int SignalingInterval
        {
            get { return signalingInterval; }
            set { signalingInterval = value; }
        }

        public SignalingSettings SignalingSettings
        {
            get
            {
                switch (signalingType)
                {
                    case SignalingType.Furioos:
                        {
                            var schema = signalingSecured ? "https" : "http";
                            return new FurioosSignalingSettings
                            (
                                url: $"{schema}://{signalingAddress}"
                            );
                        }
                    case SignalingType.WebSocket:
                        {
                            var schema = signalingSecured ? "wss" : "ws";
                            return new WebSocketSignalingSettings
                            (
                                url: $"{schema}://{signalingAddress}"
                            );
                        }
                    case SignalingType.Http:
                        {
                            var schema = signalingSecured ? "https" : "http";
                            return new HttpSignalingSettings
                            (
                                url: $"{schema}://{signalingAddress}",
                                interval: signalingInterval
                            );
                        }
                }
                throw new InvalidOperationException();
            }
        }

        public Vector2Int StreamSize
        {
            get { return streamSize; }
            set { streamSize = value; }
        }

        public VideoCodecInfo ReceiverVideoCodec
        {
            get { return receiverVideoCodec; }
            set { receiverVideoCodec = value; }
        }

        public VideoCodecInfo SenderVideoCodec
        {
            get { return senderVideoCodec; }
            set { senderVideoCodec = value; }
        }
    }
}
