using System;
using UnityEngine;
using UnityEngine.Rendering;
using UnityEngine.UI;
using UnityEngine.Events;

namespace Unity.RenderStreaming.Samples
{
    class ReceiverSample : MonoBehaviour
    {
#pragma warning disable 0649
        [SerializeField] private SignalingManager renderStreaming;
        [SerializeField] private MeshRenderer remoteVideo;
        [SerializeField] private VideoStreamReceiver receiveVideoViewer;
        [SerializeField] private SingleConnection connection;
#pragma warning restore 0649

        private string connectionId;
        private InputSender inputSender;
        private RenderStreamingSettings settings;
        private Vector2 lastSize;

        void Awake()
        {
            receiveVideoViewer.OnUpdateReceiveTexture += OnUpdateReceiveTexture;

            settings = SampleManager.Instance.Settings;

        }

        void Start()
        {
            if (renderStreaming.runOnAwake)
                return;

            if (settings != null)
                renderStreaming.useDefaultSettings = settings.UseDefaultSettings;
            if (settings?.SignalingSettings != null)
                renderStreaming.SetSignalingSettings(settings.SignalingSettings);

            renderStreaming.Run();

            Invoke(nameof(OnStart), 1f);
        }

        void OnUpdateReceiveTexture(Texture texture)
        {
            remoteVideo.material.mainTexture = texture;
        }

        private void OnStart()
        {
            if (string.IsNullOrEmpty(connectionId))
            {
                connectionId = System.Guid.NewGuid().ToString("N");
            }

            if(settings != null)
                receiveVideoViewer.SetCodec(settings.ReceiverVideoCodec);

            connection.CreateConnection(connectionId);
        }
    }
}
