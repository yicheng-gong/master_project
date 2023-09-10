using System;
using UnityEngine;
using UnityEngine.Rendering;
using UnityEngine.UI;
using UnityEngine.Events;

namespace Unity.RenderStreaming.Samples
{
    public class StartReceiver : MonoBehaviour
{
#pragma warning disable 0649
    [SerializeField] private SignalingManager renderStreaming;
    [SerializeField] private SingleConnection connection;
#pragma warning restore 0649

    private string connectionId;
    private InputSender inputSender;
    private RenderStreamingSettings settings;
    private Vector2 lastSize;

    void Awake()
    {

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
    }

    private void OnStart()
    {
        if (string.IsNullOrEmpty(connectionId))
        {
            connectionId = System.Guid.NewGuid().ToString("N");
        }

        connection.CreateConnection(connectionId);
    }
}
}
